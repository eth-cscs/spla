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
#include "pgemm_sbs/stripe_host.hpp"
#include "gemm/gemm_host.hpp"
#include "block_generation/block_cyclic_generator.hpp"
#include "mpi_util/mpi_check_status.hpp"
#include "mpi_util/mpi_match_elementary_type.hpp"
#include "spla/matrix_distribution_internal.hpp"
#include "util/blas_interface.hpp"
#include "util/common_types.hpp"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <complex>
#include <cstring>

namespace spla {

template <typename T, typename BLOCK_GEN>
StripeHost<T, BLOCK_GEN>::StripeHost(
    IntType numThreads, MPICommunicatorHandle comm,
    std::shared_ptr<Buffer<MPIAllocator>> buffer,
    std::shared_ptr<Buffer<MPIAllocator>> recvBuffer, BLOCK_GEN baseMatGen,
    ValueType alpha, const HostArrayConstView2D<ValueType> &A,
    const HostArrayConstView2D<ValueType> &B, ValueType beta,
    HostArrayView2D<ValueType> C, IntType numBlockCols)
    : state_(StripeState::Empty), localCounts_(comm.size()),
      recvDispls_(comm.size()), localRows_(comm.size()),
      localCols_(comm.size()), localRowOffsets_(comm.size()),
      localColOffsets_(comm.size()), baseMatGen_(std::move(baseMatGen)),
      buffer_(std::move(buffer)), recvBuffer_(std::move(recvBuffer)),
      comm_(std::move(comm)), numBlockCols_(numBlockCols), A_(A), B_(B), C_(C),
      alpha_(alpha), beta_(beta), numThreads_(numThreads) {
  assert(A_.dim_inner() == C.dim_inner());
  assert(buffer_);
  buffer_->resize<ValueType>(A.dim_outer() * numBlockCols *
                             baseMatGen_.max_cols_in_block());
  recvBuffer_->resize<ValueType>(A.dim_outer() * numBlockCols *
                                 baseMatGen_.max_cols_in_block());
}

template <typename T, typename BLOCK_GEN> auto StripeHost<T, BLOCK_GEN>::collect(IntType blockColIdx) -> void {
  assert(omp_get_thread_num() == 0); // only master thread should execute
  assert(blockColIdx < baseMatGen_.num_block_cols());
  if (state_.get() != StripeState::Empty) {
    throw InternalError();
  }
  // get block informations
  blockInfos_.clear(); // leaves capacity unchanged
  blockInfos_.reserve(baseMatGen_.num_block_rows() * numBlockCols_);
  for (IntType c = blockColIdx;
       c < std::min<IntType>(blockColIdx + numBlockCols_,
                             baseMatGen_.num_block_cols());
       ++c) {
    for (IntType r = 0; r < baseMatGen_.num_block_rows(); ++r) {
      blockInfos_.emplace_back(baseMatGen_.get_block_info(r, c));
    }
  }

  // calculate number of elements stored on each rank
  localRowOffsets_.assign(comm_.size(), -1);
  localColOffsets_.assign(comm_.size(), -1);
  localRows_.assign(comm_.size(), 0);
  localCols_.assign(comm_.size(), 0);
  for (const auto &info : blockInfos_) {
    assert(info.mpiRank >= 0);

    // find row of first block in stripe
    if (localRowOffsets_[info.mpiRank] < 0) {
      localRowOffsets_[info.mpiRank] = info.localRowIdx;
    }
    // find col of first block in stripe
    if (localColOffsets_[info.mpiRank] < 0) {
      localColOffsets_[info.mpiRank] = info.localColIdx;
    }

    // calculate local rows / cols by difference between first and last block
    // per rank
    localRows_[info.mpiRank] =
        info.localRowIdx - localRowOffsets_[info.mpiRank] + info.numRows;
    localCols_[info.mpiRank] =
        info.localColIdx - localColOffsets_[info.mpiRank] + info.numCols;
  }

  // compute send / recv counts
  for (IntType r = 0; r < comm_.size(); ++r) {
    localCounts_[r] = localRows_[r] * localCols_[r];
    assert(localCounts_[r] >= 0);
  }

  // Calculate displacements in receiving buffer
  recvDispls_.assign(comm_.size(), 0);
  for (IntType rank = 1; rank < comm_.size(); ++rank) {
    recvDispls_[rank] = recvDispls_[rank - 1] + localCounts_[rank - 1];
    assert(recvDispls_[rank] >= 0);
    assert(recvDispls_[rank] + localCounts_[rank] <= recvBuffer_->size<T>());
  }

  // copy into sendbuffer
  if (localCounts_[comm_.rank()]) {
    HostArrayView2D<T> sendBufferView(
        recvBuffer_->data<T>() + recvDispls_[comm_.rank()],
        localCols_[comm_.rank()], localRows_[comm_.rank()]);

    for (IntType col = 0; col < localCols_[comm_.rank()]; ++col) {
      std::memcpy(&sendBufferView(col, 0),
                  &B_(localColOffsets_[comm_.rank()] + col,
                      localRowOffsets_[comm_.rank()]),
                  sendBufferView.dim_inner() * sizeof(T));
    }
  }

  // set state atomically
  state_.set(StripeState::Collected);
}

template <typename T, typename BLOCK_GEN> auto StripeHost<T, BLOCK_GEN>::start_exchange() -> void {
  assert(omp_get_thread_num() == 0); // only master thread should execute
  if (this->state_.get() != StripeState::Collected) {
    throw InternalError();
  }

  // Exchange matrix
  mpi_check_status(MPI_Iallgatherv(
      MPI_IN_PLACE, localCounts_[comm_.rank()],
      MPIMatchElementaryType<T>::get(), recvBuffer_->data<T>(),
      localCounts_.data(), recvDispls_.data(), MPIMatchElementaryType<T>::get(),
      comm_.get(), mpiRequest_.get_and_activate()));

  this->state_.set(StripeState::InExchange);
}

template <typename T, typename BLOCK_GEN> auto StripeHost<T, BLOCK_GEN>::finalize_exchange() -> void {
  assert(omp_get_thread_num() == 0); // only master thread should execute
  if (this->state_.get() != StripeState::InExchange) {
    throw InternalError();
  }

  mpiRequest_.wait_if_active();

  this->state_.set(StripeState::Exchanged);
}

template <typename T, typename BLOCK_GEN>
auto StripeHost<T, BLOCK_GEN>::multiply() -> void {
  if (this->state_.get() != StripeState::Exchanged) {
    throw InternalError();
  }

  if (A_.size() != 0) {
    const IntType n = blockInfos_.back().globalSubColIdx -
                      blockInfos_.front().globalSubColIdx +
                      blockInfos_.back().numCols;

    // reshuffle data into full C matrix
    HostArrayView2D<T> fullStripe(buffer_->data<T>(), n, A_.dim_outer());
    const IntType stripeColOffset = blockInfos_.front().globalSubColIdx;
    for (std::size_t i = 0; i < blockInfos_.size(); ++i) {
      const auto &info = blockInfos_[i];

      assert(info.mpiRank >= 0);

      HostArrayConstView2D<T> recvDataView(
          recvBuffer_->data<T>() + recvDispls_[info.mpiRank],
          localCols_[info.mpiRank], localRows_[info.mpiRank]);

      const IntType startRow =
          info.localRowIdx - localRowOffsets_[info.mpiRank];
      const IntType startCol =
          info.localColIdx - localColOffsets_[info.mpiRank];
      for (IntType col = 0; col < info.numCols; ++col) {
        std::memcpy(&fullStripe(info.globalSubColIdx - stripeColOffset + col,
                                info.globalSubRowIdx),
                    &recvDataView(startCol + col, startRow),
                    info.numRows * sizeof(T));
      }
    }

    // multiply full C matrix.
    gemm_host<T>(numThreads_, SplaOperation::SPLA_OP_NONE,
                 SplaOperation::SPLA_OP_NONE, A_.dim_inner(), n, A_.dim_outer(),
                 alpha_, A_.data(), A_.ld_inner(), fullStripe.data(),
                 fullStripe.ld_inner(), beta_,
                 &C_(blockInfos_.front().globalSubColIdx, 0), C_.ld_inner());
  }

  this->state_.set(StripeState::Empty);
}

template class StripeHost<double, BlockCyclicGenerator>;
template class StripeHost<float, BlockCyclicGenerator>;
template class StripeHost<std::complex<double>, BlockCyclicGenerator>;
template class StripeHost<std::complex<float>, BlockCyclicGenerator>;

} // namespace spla
