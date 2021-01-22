/*
 * Copyright (c) 2020 ETH Zurich, Simon Frasch
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#include "pgemm_ssb/ring_reduce_tile_host.hpp"
#include "gemm/gemm_host.hpp"
#include "mpi_util/mpi_check_status.hpp"
#include "mpi_util/mpi_match_elementary_type.hpp"
#include "util/blas_interface.hpp"
#include "util/common_types.hpp"
#include <algorithm>
#include <cassert>
#include <complex>
#include <cstring>

namespace spla {

template <typename T>
RingReduceTileHost<T>::RingReduceTileHost(
    IntType numThreads, MPICommunicatorHandle comm,
    std::shared_ptr<Buffer<MPIAllocator>> buffer,
    std::shared_ptr<MatrixBlockGenerator> matrixDist, SplaOperation opA,
    ValueType alpha, const HostArrayConstView2D<ValueType> &A,
    const HostArrayConstView2D<ValueType> &B, ValueType beta,
    HostArrayView2D<ValueType> C)
    : matrixDist_(std::move(matrixDist)), buffer_(std::move(buffer)),
      comm_(std::move(comm)), A_(A), B_(B), C_(C), alpha_(alpha), beta_(beta),
      opA_(opA), numThreads_(numThreads) {
  assert(A_.dim_inner() == B_.dim_inner());
  assert(buffer_);
  assert(opA_ == SplaOperation::SPLA_OP_CONJ_TRANSPOSE ||
         opA_ == SplaOperation::SPLA_OP_TRANSPOSE);
  const auto blockSize =
      matrixDist_->max_cols_in_block() * matrixDist_->max_rows_in_block();
  buffer_->resize<ValueType>(2 * blockSize);
  sendView_ =
      HostArrayView2D<T>(buffer_->data<T>(), matrixDist_->max_cols_in_block(),
                         matrixDist_->max_rows_in_block());
  recvView_ = HostArrayView2D<T>(buffer_->data<T>() + blockSize,
                                 matrixDist_->max_cols_in_block(),
                                 matrixDist_->max_rows_in_block());
}

template <typename T>
auto RingReduceTileHost<T>::prepare(IntType blockRowIdx, IntType blockColIdx,
                                    IntType numBlockRows, IntType numBlockCols)
    -> void {
  blockInfos_.resize(numBlockRows * numBlockCols);

  numBlockRows_ = numBlockRows;
  numBlockCols_ = numBlockCols;
  myBlockIdx_ = -1;
  currentBlockIdx = 0;

  // compute block infos
  for (IntType ic = 0; ic < numBlockCols; ++ic) {
    for (IntType ir = 0; ir < numBlockRows; ++ir) {
      blockInfos_[ic * numBlockRows + ir] =
          matrixDist_->get_block_info(blockRowIdx + ir, blockColIdx + ic);
      if (blockInfos_[ic * numBlockRows + ir].mpiRank == comm_.rank()) {
        assert(myBlockIdx_ <
               0); // each rank must receive at most one block within grid
        myBlockIdx_ = ic * numBlockRows + ir;
      }
    }
  }

  std::memset(recvView_.data(), 0, recvView_.size() * sizeof(T));

}

template <typename T> auto RingReduceTileHost<T>::process_step() -> bool {
  const IntType numBlocks = numBlockRows_ * numBlockCols_;
  const bool accumulateRequired = numBlocks != comm_.size();

  if (currentBlockIdx < numBlocks) {
    if (!accumulateRequired) {
      const IntType startGridIdx = (myBlockIdx_ + 1) % numBlocks;

      const int sendRank =
          blockInfos_[myBlockIdx_ == 0 ? numBlocks - 1 : myBlockIdx_ - 1]
              .mpiRank;
      const int recvRank = blockInfos_[(myBlockIdx_ + 1) % numBlocks].mpiRank;

      const IntType gridBlockIdx = (startGridIdx + currentBlockIdx) % numBlocks;
      const auto &info = blockInfos_[gridBlockIdx];
      assert(info.mpiRank >= 0); // Mirror distribution not supported

      sendReq_.wait_if_active();
      recvReq_.wait_if_active();
      std::swap(sendView_, recvView_);

      if (currentBlockIdx < numBlocks - 1) {
        MPI_Irecv(recvView_.data(), recvView_.size(),
                  MPIMatchElementaryType<T>::get(), recvRank, 0, comm_.get(),
                  recvReq_.get_and_activate());
      }
      if (A_.dim_inner() != 0) {
        gemm_host<T>(numThreads_, opA_, SplaOperation::SPLA_OP_NONE,
                     info.numRows, info.numCols, A_.dim_inner(), alpha_,
                     &A_(info.globalSubRowIdx, 0), A_.ld_inner(),
                     &B_(info.globalSubColIdx, 0), B_.ld_inner(), 1.0,
                     sendView_.data(), sendView_.ld_inner());
      }
      if (currentBlockIdx < numBlocks - 1) // continue sending around in ring
        MPI_Send(sendView_.data(), sendView_.size(),
                 MPIMatchElementaryType<T>::get(), sendRank, 0, comm_.get());
    } else {
      assert(accumulateRequired);

      const BlockInfo &info = blockInfos_[currentBlockIdx];

      sendReq_.wait_if_active();

      if (A_.dim_inner() == 0) {
        std::memset(recvView_.data(), 0, recvView_.size() * sizeof(T));
      } else {
        gemm_host<T>(numThreads_, opA_, SplaOperation::SPLA_OP_NONE,
                     info.numRows, info.numCols, A_.dim_inner(), alpha_,
                     &A_(info.globalSubRowIdx, 0), A_.ld_inner(),
                     &B_(info.globalSubColIdx, 0), B_.ld_inner(), 0.0,
                     recvView_.data(), recvView_.ld_inner());
      }
      mpi_check_status(MPI_Ireduce(
          recvView_.data(), sendView_.data(), sendView_.size(),
          MPIMatchElementaryType<ValueType>::get(), MPI_SUM, info.mpiRank,
          comm_.get(), sendReq_.get_and_activate()));

    }

  } else if (currentBlockIdx == numBlocks) {
    // add tile to result as final step

    sendReq_.wait_if_active();
    recvReq_.wait_if_active();

    if (myBlockIdx_ >= 0) {
      const auto &myInfo = blockInfos_[myBlockIdx_];
      for (IntType col = 0; col < myInfo.numCols; ++col) {
        for (IntType row = 0; row < myInfo.numRows; ++row) {
          C_(myInfo.localColIdx + col, myInfo.localRowIdx + row) =
              beta_ * C_(myInfo.localColIdx + col, myInfo.localRowIdx + row) +
              sendView_(col, row);
        }
      }
    }
  }

  ++currentBlockIdx;
  return currentBlockIdx <= numBlocks;
}

template class RingReduceTileHost<double>;
template class RingReduceTileHost<float>;
template class RingReduceTileHost<std::complex<double>>;
template class RingReduceTileHost<std::complex<float>>;

} // namespace spla
