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
#include "pgemm_sbs/stripe_gpu.hpp"

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstring>

#include "block_generation/block_cyclic_generator.hpp"
#include "gpu_util/multiply_gpu.hpp"
#include "mpi_util/mpi_check_status.hpp"
#include "mpi_util/mpi_match_elementary_type.hpp"
#include "util/common_types.hpp"
namespace spla {

template <typename T, typename BLOCK_GEN>
StripeGPU<T, BLOCK_GEN>::StripeGPU(MPICommunicatorHandle comm, GPUBlasHandle blasHandle,
                                   std::shared_ptr<Buffer<PinnedAllocator>> buffer,
                                   std::shared_ptr<Buffer<PinnedAllocator>> recvBuffer,
                                   std::shared_ptr<Buffer<GPUAllocator>> bufferGPU,
                                   IntType maxGPUStripeSize, BLOCK_GEN baseMatGen, ValueType alpha,
                                   GPUMatrixAccessor<GPUArrayConstView2D<T>> A,
                                   HostArrayConstView2D<ValueType> matBViewHost,
                                   GPUArrayConstView2D<ValueType> matBViewGPU, ValueType beta,
                                   GPUMatrixAccessor<GPUArrayView2D<T>> C,
                                   HostArrayView2D<T> viewCHost, IntType numBlockCols)
    : state_(StripeState::Empty),
      localCounts_(comm.size()),
      recvDispls_(comm.size()),
      localRows_(comm.size()),
      localCols_(comm.size()),
      localRowOffsets_(comm.size()),
      localColOffsets_(comm.size()),
      baseMatGen_(std::move(baseMatGen)),
      buffer_(std::move(buffer)),
      recvBuffer_(std::move(recvBuffer)),
      bufferGPU_(std::move(bufferGPU)),
      comm_(std::move(comm)),
      blasHandle_(std::move(blasHandle)),
      maxGPUStripeSize_(maxGPUStripeSize),
      numBlockCols_(numBlockCols),
      matA_(A),
      matC_(C),
      viewCHost_(viewCHost),
      matBViewHost_(matBViewHost),
      matBViewGPU_(matBViewGPU),
      alpha_(alpha),
      beta_(beta) {
  // assert(.dim_inner() == C.dim_inner());
  // assert(buffer_);
  buffer_->resize<ValueType>(A.cols() * numBlockCols * baseMatGen_.max_cols_in_block());
  recvBuffer_->resize<ValueType>(A.cols() * numBlockCols * baseMatGen_.max_cols_in_block());
}

template <typename T, typename BLOCK_GEN>
auto StripeGPU<T, BLOCK_GEN>::collect(IntType blockColIdx) -> void {
  assert(state_.get() == StripeState::Empty);
  assert(blockColIdx < baseMatGen_.num_block_cols());
  if (state_.get() != StripeState::Empty) {
    throw InternalError();
  }
  // get block informations
  blockInfos_.clear();  // leaves capacity unchanged
  blockInfos_.reserve(baseMatGen_.num_block_rows() * numBlockCols_);
  for (IntType c = blockColIdx;
       c < std::min<IntType>(blockColIdx + numBlockCols_, baseMatGen_.num_block_cols()); ++c) {
    for (IntType r = 0; r < baseMatGen_.num_block_rows(); ++r) {
      blockInfos_.emplace_back(baseMatGen_.get_block_info(r, c));
    }
  }

  // calculate number of elements stored on each rank
  localRowOffsets_.assign(comm_.size(), -1);
  localColOffsets_.assign(comm_.size(), -1);
  for (const auto& info : blockInfos_) {
    assert(info.mpiRank >= 0);

    // find row of first block in stripe
    if (localRowOffsets_[info.mpiRank] < 0) {
      localRowOffsets_[info.mpiRank] = info.localRowIdx;
    }
    // find col of first block in stripe
    if (localColOffsets_[info.mpiRank] < 0) {
      localColOffsets_[info.mpiRank] = info.localColIdx;
    }

    // calculate local rows / cols by difference between first and last block per rank
    localRows_[info.mpiRank] = info.localRowIdx - localRowOffsets_[info.mpiRank] + info.numRows;
    localCols_[info.mpiRank] = info.localColIdx - localColOffsets_[info.mpiRank] + info.numCols;
  }

  // compute send / recv counts
  for (IntType r = 0; r < comm_.size(); ++r) {
    localCounts_[r] = localRows_[r] * localCols_[r];
  }

  // Calculate displacements in receiving buffer
  recvDispls_.assign(comm_.size(), 0);
  for (IntType rank = 1; rank < comm_.size(); ++rank) {
    recvDispls_[rank] = recvDispls_[rank - 1] + localCounts_[rank - 1];
  }

  // copy into sendbuffer
  if (localCounts_[comm_.rank()]) {
    HostArrayView2D<T> sendBufferView(recvBuffer_->data<T>() + recvDispls_[comm_.rank()],
                                      localCols_[comm_.rank()], localRows_[comm_.rank()]);
    if (!matBViewHost_.empty()) {
      for (IntType col = 0; col < localCols_[comm_.rank()]; ++col) {
        std::memcpy(
            &sendBufferView(col, 0),
            &matBViewHost_(localColOffsets_[comm_.rank()] + col, localRowOffsets_[comm_.rank()]),
            sendBufferView.dim_inner() * sizeof(T));
      }
    } else {
      GPUArrayConstView2D<T> subMatBView(
          matBViewGPU_.data() +
              matBViewGPU_.index(localColOffsets_[comm_.rank()], localRowOffsets_[comm_.rank()]),
          localCols_[comm_.rank()], localRows_[comm_.rank()], matBViewGPU_.ld_inner());

      copy_from_gpu_async(blasHandle_.stream_handle().get(), subMatBView, sendBufferView);
    }
  }

  // set state atomically
  state_.set(StripeState::Collected);
}

template <typename T, typename BLOCK_GEN>
auto StripeGPU<T, BLOCK_GEN>::start_exchange() -> void {
  assert(this->state_.get() == StripeState::Collected);
  if (this->state_.get() != StripeState::Collected) {
    throw InternalError();
  }

  // wait transfer to gpu to finish
  gpu::check_status(gpu::stream_synchronize(blasHandle_.stream_handle().get()));

  // Exchange matrix
  mpi_check_status(
      MPI_Iallgatherv(MPI_IN_PLACE, localCounts_[comm_.rank()], MPIMatchElementaryType<T>::get(),
                      recvBuffer_->data<T>(), localCounts_.data(), recvDispls_.data(),
                      MPIMatchElementaryType<T>::get(), comm_.get(), request_.get_and_activate()));

  // set state atomically
  this->state_.set(StripeState::InExchange);
}

template <typename T, typename BLOCK_GEN>
auto StripeGPU<T, BLOCK_GEN>::finalize_exchange() -> void {
  assert(state_.get() == StripeState::InExchange);
  if (state_.get() != StripeState::InExchange) {
    throw InternalError();
  }
  request_.wait_if_active();

  // set state atomically
  state_.set(StripeState::Exchanged);
}

template <typename T, typename BLOCK_GEN>
auto StripeGPU<T, BLOCK_GEN>::multiply() -> void {
  assert(this->state_.get() == StripeState::Exchanged);
  if (this->state_.get() != StripeState::Exchanged) {
    throw InternalError();
  }

  if (matA_.size()) {
    const IntType n = blockInfos_.back().globalSubColIdx - blockInfos_.front().globalSubColIdx +
                      blockInfos_.back().numCols;
    const IntType m = matA_.rows();

    // reshuffle data into full C matrix
    HostArrayView2D<T> fullStripe(buffer_->data<T>(), n, matA_.cols());
    const IntType stripeColOffset = blockInfos_.front().globalSubColIdx;
    for (const auto& info : blockInfos_) {
      assert(info.mpiRank >= 0);

      HostArrayConstView2D<T> recvDataView(recvBuffer_->data<T>() + recvDispls_[info.mpiRank],
                                           localCols_[info.mpiRank], localRows_[info.mpiRank]);

      const IntType startRow = info.localRowIdx - localRowOffsets_[info.mpiRank];
      const IntType startCol = info.localColIdx - localColOffsets_[info.mpiRank];
      for (IntType col = 0; col < info.numCols; ++col) {
        std::memcpy(&fullStripe(info.globalSubColIdx - stripeColOffset + col, info.globalSubRowIdx),
                    &recvDataView(startCol + col, startRow), info.numRows * sizeof(T));
      }
    }

    IntType rowBlockSize = std::min<IntType>(std::sqrt(matC_.max_tile_size()), m);
    IntType colBlockSize = std::min<IntType>(matC_.max_tile_size() / rowBlockSize, n);
    rowBlockSize = std::min<IntType>(matC_.max_tile_size() / colBlockSize, m);

    GPUMatrixAccessor<GPUArrayConstView2D<T>> matB(fullStripe, maxGPUStripeSize_, bufferGPU_);

    for (IntType col = 0; col < n; col += colBlockSize) {
      const IntType currentColBlockSize = std::min<IntType>(n - col, colBlockSize);
      for (IntType row = 0; row < m; row += rowBlockSize) {
        const IntType currentRowBlockSize = std::min<IntType>(m - row, rowBlockSize);
        auto viewC = matC_.get_tile(0, blockInfos_.front().globalSubColIdx, currentRowBlockSize,
                                    currentColBlockSize, blasHandle_.stream_handle().get());
        multiply_gpu<T>(blasHandle_.get(), gpu::blas::operation::None, gpu::blas::operation::None,
                        alpha_, matA_, matB, beta_, viewC);
        if (!viewCHost_.empty()) {
          copy_from_gpu_async(
              blasHandle_.stream_handle().get(), GPUArrayConstView2D<T>(viewC),
              HostArrayView2D<T>(&viewCHost_(blockInfos_.front().globalSubColIdx, 0),
                                 viewC.dim_outer(), viewC.dim_inner(), viewCHost_.ld_inner()));
        }
      }
    }
  }

  // set state atomically
  this->state_.set(StripeState::Empty);
}

template class StripeGPU<double, BlockCyclicGenerator>;
template class StripeGPU<float, BlockCyclicGenerator>;
template class StripeGPU<gpu::blas::ComplexFloatType, BlockCyclicGenerator>;
template class StripeGPU<gpu::blas::ComplexDoubleType, BlockCyclicGenerator>;

}  // namespace spla
