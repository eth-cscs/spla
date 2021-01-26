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
#include <vector>

namespace spla {

static constexpr int resultTag = 1;
static constexpr int ringTag = 2;

template <typename T>
RingReduceTileHost<T>::RingReduceTileHost(
    IntType numThreads, MPICommunicatorHandle comm,
    std::shared_ptr<Buffer<MPIAllocator>> buffer,
    std::shared_ptr<Buffer<MPIAllocator>> resultBuffer,
    std::shared_ptr<MatrixBlockGenerator> matrixDist, SplaOperation opA,
    ValueType alpha, const HostArrayConstView2D<ValueType> &A,
    const HostArrayConstView2D<ValueType> &B, ValueType beta,
    HostArrayView2D<ValueType> C)
    : state_(TileState::Empty), matrixDist_(std::move(matrixDist)),
      buffer_(std::move(buffer)), resultBuffer_(std::move(resultBuffer)),
      comm_(std::move(comm)), A_(A), B_(B), C_(C), alpha_(alpha), beta_(beta),
      opA_(opA), numThreads_(numThreads) {
  assert(A_.dim_inner() == B_.dim_inner());
  assert(buffer_);
  assert(opA_ == SplaOperation::SPLA_OP_CONJ_TRANSPOSE ||
         opA_ == SplaOperation::SPLA_OP_TRANSPOSE);
  const auto maxBlockSize =
      matrixDist_->max_cols_in_block() * matrixDist_->max_rows_in_block();
  buffer_->resize<ValueType>(2 * maxBlockSize);
  sendView_ = HostArrayView1D<T>(buffer_->data<T>(), maxBlockSize);
  recvView_ = HostArrayView1D<T>(buffer_->data<T>() + maxBlockSize, maxBlockSize);
}

template <typename T>
auto RingReduceTileHost<T>::prepare(
    std::vector<BlockInfo>::const_iterator begin,
    std::vector<BlockInfo>::const_iterator end) -> void {
  assert(state_ == TileState::Empty);

  numMyBlocksReduced_ = 0;

  blockInfos_.assign(begin, end);

  currentBlockIdx = 0;
  const IntType rankOffset = blockInfos_.front().mpiRank;
  myStartIdx_ = (rankOffset + comm_.rank()) % blockInfos_.size();
  sendRank_ = comm_.rank() == 0 ? comm_.size() - 1 : comm_.rank() - 1;
  recvRank_ = (comm_.rank() + 1) % comm_.size();

  myBlockIndices_.resize(0);
  for (IntType i = 0; i < blockInfos_.size(); ++i) {
    if (blockInfos_[i].mpiRank == comm_.rank()) {
      myBlockIndices_.emplace_back(i);
    }
  }

  const auto maxBlockSize =
      matrixDist_->max_cols_in_block() * matrixDist_->max_rows_in_block();


  std::memset(recvView_.data(), 0, recvView_.size() * sizeof(T));

  const bool accumulateRequired = blockInfos_.size() != comm_.size();
  resultBuffer_->resize<T>(std::max<std::size_t>(myBlockIndices_.size(), 1) *
                           maxBlockSize);
  resultRecvs_.resize(myBlockIndices_.size());

  if (!accumulateRequired) {
    for(IntType i = 0; i < myBlockIndices_.size(); ++i){
      // Determine rank to receive result from by computing the rank, which
      // holds the block initially and substracting the number of steps in the
      // ring (blocks are send backwards)
      const auto originRank =
          (myBlockIndices_[i] + 2 * comm_.size() - rankOffset - (blockInfos_.size() - 1)) % comm_.size();
      const auto& info = blockInfos_[myBlockIndices_[i]];
      // Post receive for each block this ranks requires
      MPI_Irecv(resultBuffer_->data<T>() + i * maxBlockSize,
                info.numCols * info.numRows, MPIMatchElementaryType<T>::get(),
                originRank, resultTag, comm_.get(),
                resultRecvs_[i].get_and_activate());
    }
  } else {
    std::memset(resultBuffer_->data<T>(), 0, maxBlockSize * sizeof(T));
  }

  state_ = TileState::Prepared;
}

template <typename T> auto RingReduceTileHost<T>::process_step() -> bool {
  state_ = TileState::PartiallyProcessed;
  const bool accumulateRequired = blockInfos_.size() != comm_.size();
  const IntType numBlocks = blockInfos_.size();
  const auto maxBlockSize =
      matrixDist_->max_cols_in_block() * matrixDist_->max_rows_in_block();

  if (currentBlockIdx < numBlocks) {
    if (!accumulateRequired) {

      const auto &info = blockInfos_[(myStartIdx_ + currentBlockIdx) % blockInfos_.size()];
      const auto &nextInfo = blockInfos_[(myStartIdx_ + currentBlockIdx + 1) % blockInfos_.size()];
      assert(info.mpiRank >= 0); // Mirror distribution not supported

      sendReq_.wait_if_active();
      recvReq_.wait_if_active();
      std::swap(sendView_, recvView_);

      if (currentBlockIdx < numBlocks - 1) {
        MPI_Irecv(recvView_.data(), nextInfo.numCols * nextInfo.numRows,
                  MPIMatchElementaryType<T>::get(), recvRank_, ringTag,
                  comm_.get(), recvReq_.get_and_activate());
      }
      if (A_.dim_inner() != 0) {
        gemm_host<T>(numThreads_, opA_, SplaOperation::SPLA_OP_NONE,
                     info.numRows, info.numCols, A_.dim_inner(), alpha_,
                     &A_(info.globalSubRowIdx, 0), A_.ld_inner(),
                     &B_(info.globalSubColIdx, 0), B_.ld_inner(), 1.0,
                     sendView_.data(), info.numRows);
      }
      if (currentBlockIdx < numBlocks - 1) { // continue sending around in ring
        MPI_Isend(sendView_.data(), info.numRows * info.numCols,
                  MPIMatchElementaryType<T>::get(), sendRank_, ringTag,
                  comm_.get(), sendReq_.get_and_activate());
      } else {
        MPI_Isend(sendView_.data(), info.numRows * info.numCols,
                  MPIMatchElementaryType<T>::get(), info.mpiRank, resultTag,
                  comm_.get(), sendReq_.get_and_activate());
      }
    } else {
      assert(accumulateRequired);

      const BlockInfo &info = blockInfos_[currentBlockIdx];

      sendReq_.wait_if_active();

      if (A_.dim_inner() == 0) {
        std::memset(sendView_.data(), 0, sendView_.size() * sizeof(T));
      } else {
        gemm_host<T>(numThreads_, opA_, SplaOperation::SPLA_OP_NONE,
                     info.numRows, info.numCols, A_.dim_inner(), alpha_,
                     &A_(info.globalSubRowIdx, 0), A_.ld_inner(),
                     &B_(info.globalSubColIdx, 0), B_.ld_inner(), 0.0,
                     sendView_.data(), info.numRows);
      }
      mpi_check_status(MPI_Ireduce(
          sendView_.data(),
          resultBuffer_->data<T>() + numMyBlocksReduced_ * maxBlockSize,
          info.numCols * info.numRows, MPIMatchElementaryType<ValueType>::get(),
          MPI_SUM, info.mpiRank, comm_.get(), sendReq_.get_and_activate()));
      if(info.mpiRank == comm_.rank()) ++numMyBlocksReduced_;

    }

  } else if (currentBlockIdx == numBlocks) {
    // add tile to result as final step

    sendReq_.wait_if_active();
    recvReq_.wait_if_active();

    for(IntType i = 0; i < myBlockIndices_.size(); ++i){
      resultRecvs_[i].wait_if_active();
      const auto &info = blockInfos_[myBlockIndices_[i]];
      HostArrayView2D<T> resultView(resultBuffer_->data<T>() + i * maxBlockSize,
                                    info.numCols, info.numRows);
      for (IntType col = 0; col < info.numCols; ++col) {
        for (IntType row = 0; row < info.numRows; ++row) {
          C_(info.localColIdx + col, info.localRowIdx + row) =
              beta_ * C_(info.localColIdx + col, info.localRowIdx + row) +
              resultView(col, row);
        }
      }
    }

    state_ = TileState::Empty;
  }

  ++currentBlockIdx;
  return currentBlockIdx <= numBlocks;
}

template class RingReduceTileHost<double>;
template class RingReduceTileHost<float>;
template class RingReduceTileHost<std::complex<double>>;
template class RingReduceTileHost<std::complex<float>>;

} // namespace spla
