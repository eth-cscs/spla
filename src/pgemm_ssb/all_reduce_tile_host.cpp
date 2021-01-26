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
#include "pgemm_ssb/all_reduce_tile_host.hpp"
#include "pgemm_ssb/add_kernel.hpp"
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
AllReduceTileHost<T>::AllReduceTileHost(IntType numThreads, MPICommunicatorHandle comm,
                      std::shared_ptr<Buffer<MPIAllocator>> buffer,
                      std::shared_ptr<MatrixBlockGenerator> matrixDist,
                      SplaOperation opA, ValueType alpha,
                      const HostArrayConstView2D<ValueType> &A,
                      const HostArrayConstView2D<ValueType> &B, ValueType beta,
                      HostArrayView2D<ValueType> C, IntType numBlockRows,
                      IntType numBlockCols)
    : state_(TileState::Empty), matrixDist_(std::move(matrixDist)),
      buffer_(std::move(buffer)), comm_(std::move(comm)),
      numBlockRows_(numBlockRows), numBlockCols_(numBlockCols), A_(A), B_(B),
      C_(C), alpha_(alpha), beta_(beta), opA_(opA), numThreads_(numThreads) {
  assert(A_.dim_inner() == B_.dim_inner());
  assert(buffer_);
  assert(opA_ == SplaOperation::SPLA_OP_CONJ_TRANSPOSE ||
         opA_ == SplaOperation::SPLA_OP_TRANSPOSE);
  buffer_->resize<ValueType>(numBlockRows * numBlockCols *
                             matrixDist_->max_rows_in_block() *
                             matrixDist_->max_cols_in_block());
}

template <typename T>
auto AllReduceTileHost<T>::multiply(IntType blockRowIdx, IntType blockColIdx) -> void {
  assert(blockRowIdx < matrixDist_->num_block_rows());
  assert(blockColIdx < matrixDist_->num_block_cols());

  if (state_.get() != TileState::Empty) {
    throw InternalError();
  }
  const IntType numCurrentBlockRows =
      std::min(matrixDist_->num_block_rows() - blockRowIdx, numBlockRows_);
  const IntType numCurrentBlockCols =
      std::min(matrixDist_->num_block_cols() - blockColIdx, numBlockCols_);

  // get block informations
  blockInfos_.clear(); // leaves capacity unchanged
  blockInfos_.reserve(numCurrentBlockRows * numCurrentBlockCols);
  for (IntType c = 0; c < numCurrentBlockCols; ++c) {
    for (IntType r = 0; r < numCurrentBlockRows; ++r) {
      blockInfos_.emplace_back(
          matrixDist_->get_block_info(blockRowIdx + r, blockColIdx + c));
    }
  }

  const IntType numTileRows = blockInfos_.back().numRows +
                              blockInfos_.back().globalSubRowIdx -
                              blockInfos_.front().globalSubRowIdx;
  const IntType numTileCols = blockInfos_.back().numCols +
                              blockInfos_.back().globalSubColIdx -
                              blockInfos_.front().globalSubColIdx;

  tile_ = HostArrayView2D<ValueType>(buffer_->data<ValueType>(), numTileCols,
                                     numTileRows);
  state_.set(TileState::Prepared);

  if (A_.dim_inner() == 0) {
    std::memset(tile_.data(), 0, tile_.size() * sizeof(ValueType));
  } else {
    const IntType lda = A_.ld_inner();
    const IntType ldb = B_.ld_inner();
    const IntType ldc = tile_.ld_inner();

    const IntType k = A_.dim_inner();

    const ValueType beta = 0.0;

    gemm_host<T>(numThreads_, opA_, SplaOperation::SPLA_OP_NONE,
                 tile_.dim_inner(), tile_.dim_outer(), k, alpha_,
                 &A_(blockInfos_.front().globalSubRowIdx, 0), lda,
                 &B_(blockInfos_.front().globalSubColIdx, 0), ldb, beta,
                 tile_.data(), ldc);
  }


  state_.set(TileState::Multiplied);
}

template <typename T> auto AllReduceTileHost<T>::start_exchange() -> void {
  assert(omp_get_thread_num() == 0); // only master thread should execute
  if (this->state_.get() != TileState::Multiplied) {
    throw InternalError();
  }

  if (blockInfos_.size() == 1 && blockInfos_.front().mpiRank >= 0) {
    const auto &info = blockInfos_.front();
    // result is send to single rank
    if (comm_.rank() == info.mpiRank) {
      mpi_check_status(MPI_Ireduce(
          MPI_IN_PLACE, this->tile_.data(), this->tile_.size(),
          MPIMatchElementaryType<ValueType>::get(), MPI_SUM, info.mpiRank,
          comm_.get(), mpiRequest_.get_and_activate()));
    } else {
      mpi_check_status(MPI_Ireduce(
          this->tile_.data(), nullptr, this->tile_.size(),
          MPIMatchElementaryType<ValueType>::get(), MPI_SUM, info.mpiRank,
          comm_.get(), mpiRequest_.get_and_activate()));
    }
  } else {
    // result required on all ranks
    mpi_check_status(
        MPI_Iallreduce(MPI_IN_PLACE, this->tile_.data(), this->tile_.size(),
                       MPIMatchElementaryType<ValueType>::get(), MPI_SUM,
                       comm_.get(), mpiRequest_.get_and_activate()));
  }

  this->state_.set(TileState::InExchange);
}

template <typename T> auto AllReduceTileHost<T>::finalize_exchange() -> void {
  assert(omp_get_thread_num() == 0); // only master thread should execute
  if (this->state_.get() != TileState::InExchange) {
    throw InternalError();
  }

  mpiRequest_.wait_if_active();

  this->state_.set(TileState::Exchanged);
}

template <typename T> auto AllReduceTileHost<T>::extract() -> void {
  if (this->state_.get() != TileState::Exchanged) {
    throw InternalError();
  }

  // iterate over all blocks within tile
  for (const auto &info : blockInfos_) {
    const IntType tileRowOffset =
        info.globalSubRowIdx - blockInfos_.front().globalSubRowIdx;
    const IntType tileColOffset =
        info.globalSubColIdx - blockInfos_.front().globalSubColIdx;

    if (info.mpiRank == comm_.rank() || info.mpiRank < 0) {
      add_kernel(info.numRows, info.numCols,
                 &(tile_(tileColOffset, tileRowOffset)), tile_.ld_inner(),
                 beta_, &(C_(info.localColIdx, info.localRowIdx)),
                 C_.ld_inner());
    }
  }
  this->state_.set(TileState::Empty);
}

template class AllReduceTileHost<double>;
template class AllReduceTileHost<float>;
template class AllReduceTileHost<std::complex<double>>;
template class AllReduceTileHost<std::complex<float>>;

} // namespace spla
