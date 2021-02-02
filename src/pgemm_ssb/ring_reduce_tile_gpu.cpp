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
#include "pgemm_ssb/ring_reduce_tile_gpu.hpp"
#include "block_generation/matrix_block_generator.hpp"
#include "gpu_util/gpu_blas_api.hpp"
#include "gpu_util/gpu_helper.hpp"
#include "gpu_util/gpu_runtime_api.hpp"
#include "gpu_util/gpu_transfer.hpp"
#include "gpu_util/multiply_gpu.hpp"
#include "memory/gpu_array_const_view.hpp"
#include "memory/host_array_view.hpp"
#include "mpi_util/mpi_check_status.hpp"
#include "mpi_util/mpi_match_elementary_type.hpp"
#include "pgemm_ssb/add_kernel.hpp"
#include "util/blas_interface.hpp"
#include "util/common_types.hpp"
#include <algorithm>
#include <cassert>
#include <complex>
#include <cstring>
#include <memory>

namespace spla {

static constexpr int resultTag = 1;
static constexpr int ringTag = 2;

static auto call_gpu_geam(const gpu::blas::HandleType &handle,
                          const gpu::blas::OperationType &transa,
                          const gpu::blas::OperationType &transb, int m, int n,
                          float alpha, const float *A, int lda, float beta,
                          const float *B, int ldb, float *C, int ldc) -> void {
  gpu::blas::sgeam(handle, transa, transb, m, n, &alpha, A, lda, &beta, B, ldb,
                   C, ldc);
}

static auto call_gpu_geam(const gpu::blas::HandleType &handle,
                          const gpu::blas::OperationType &transa,
                          const gpu::blas::OperationType &transb, int m, int n,
                          double alpha, const double *A, int lda, double beta,
                          const double *B, int ldb, double *C, int ldc)
    -> void {
  gpu::blas::dgeam(handle, transa, transb, m, n, &alpha, A, lda, &beta, B, ldb,
                   C, ldc);
}

static auto call_gpu_geam(const gpu::blas::HandleType &handle,
                          const gpu::blas::OperationType &transa,
                          const gpu::blas::OperationType &transb, int m, int n,
                          gpu::blas::ComplexDoubleType alpha,
                          const gpu::blas::ComplexDoubleType *A, int lda,
                          gpu::blas::ComplexDoubleType beta,
                          const gpu::blas::ComplexDoubleType *B, int ldb,
                          gpu::blas::ComplexDoubleType *C, int ldc) -> void {
  gpu::blas::zgeam(handle, transa, transb, m, n, &alpha, A, lda, &beta, B, ldb,
                   C, ldc);
}

static auto call_gpu_geam(const gpu::blas::HandleType &handle,
                          const gpu::blas::OperationType &transa,
                          const gpu::blas::OperationType &transb, int m, int n,
                          gpu::blas::ComplexFloatType alpha,
                          const gpu::blas::ComplexFloatType *A, int lda,
                          gpu::blas::ComplexFloatType beta,
                          const gpu::blas::ComplexFloatType *B, int ldb,
                          gpu::blas::ComplexFloatType *C, int ldc) -> void {
  gpu::blas::cgeam(handle, transa, transb, m, n, &alpha, A, lda, &beta, B, ldb,
                   C, ldc);
}

template <typename T>
RingReduceTileGPU<T>::RingReduceTileGPU(
    MPICommunicatorHandle comm, std::vector<RingBlock<T>> ringBlocks,
    std::shared_ptr<Buffer<PinnedAllocator>> resultBufferHost,
    std::shared_ptr<MatrixBlockGenerator> matrixDist, SplaOperation opA,
    ValueType alpha, ValueType beta, HostArrayView2D<ValueType> HostMatC,
    GPUArrayView2D<ValueType> GPUMatC)
    : state_(TileState::Empty), comm_(std::move(comm)),
      matrixDist_(std::move(matrixDist)), ringBlocks_(std::move(ringBlocks)),
      resultBufferHost_(std::move(resultBufferHost)), HostMatC_(HostMatC),
      GPUMatC_(GPUMatC), alpha_(alpha), beta_(beta), opA_(opA) {

  assert(resultBufferHost_);
  assert(opA_ == SplaOperation::SPLA_OP_CONJ_TRANSPOSE ||
         opA_ == SplaOperation::SPLA_OP_TRANSPOSE);
}

template <typename T>
auto RingReduceTileGPU<T>::prepare(std::vector<BlockInfo>::const_iterator begin,
                                   std::vector<BlockInfo>::const_iterator end)
    -> void {
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
    assert(blockInfos_[i].mpiRank >= 0); // Mirror distribution not supported
    if (blockInfos_[i].mpiRank == comm_.rank()) {
      myBlockIndices_.emplace_back(i);
    }
  }

  const auto maxBlockSize =
      matrixDist_->max_cols_in_block() * matrixDist_->max_rows_in_block();

  for (auto &b : ringBlocks_) {
    gpu::check_status(
        gpu::stream_synchronize(b.blasHandle.stream_handle().get()));
    std::memset(b.tileViewHost.data(), 0, b.tileViewHost.size() * sizeof(T));
  }

  const bool accumulateRequired = blockInfos_.size() != comm_.size();

  const std::size_t resultSize =
      std::max<std::size_t>(myBlockIndices_.size(), 1) * maxBlockSize;
  if (resultBufferHost_->size<T>() < resultSize) {
    resultBufferHost_->resize<T>(resultSize);
  }

  resultRecvs_.resize(myBlockIndices_.size());

  if (!accumulateRequired) {
    for (IntType i = 0; i < myBlockIndices_.size(); ++i) {
      // Determine rank to receive result from by computing the rank, which
      // holds the block initially and substracting the number of steps in the
      // ring (blocks are send backwards)
      const auto originRank = (myBlockIndices_[i] + 2 * comm_.size() -
                               rankOffset - (blockInfos_.size() - 1)) %
                              comm_.size();
      const auto &info = blockInfos_[myBlockIndices_[i]];
      // Post receive for each block this ranks requires
      MPI_Irecv(resultBufferHost_->data<T>() + i * maxBlockSize,
                info.numCols * info.numRows, MPIMatchElementaryType<T>::get(),
                originRank, resultTag, comm_.get(),
                resultRecvs_[i].get_and_activate());
    }
  } else {
    std::memset(resultBufferHost_->data<T>(), 0,
                resultBufferHost_->size<T>() * sizeof(T));
  }

  // Start processing of first blocks
  if (ringBlocks_.front().matA.rows() != 0) {
    for (IntType i = 0;
         i < std::min<IntType>(ringBlocks_.size(), blockInfos_.size()); ++i) {
      const auto &info =
          accumulateRequired
              ? blockInfos_[i]
              : blockInfos_[(myStartIdx_ + i) % blockInfos_.size()];
      auto& block = ringBlocks_[i];
      ValueType beta = RealValueGPU<T>::create(0.0);

      auto opAGPU = opA_ == SplaOperation::SPLA_OP_TRANSPOSE
                        ? gpu::blas::operation::Transpose
                        : gpu::blas::operation::ConjugateTranspose;
      multiply_gpu<ValueType>(
          block.blasHandle.get(), opAGPU, gpu::blas::operation::None, alpha_,
          block.matA.sub_accessor(0, info.globalSubRowIdx, block.matA.rows(),
                             info.numRows),
          block.matB.sub_accessor(0, info.globalSubColIdx, block.matB.rows(),
                             info.numCols),
          beta,
          GPUArrayView2D<T>(block.tileViewGPU.data(), info.numCols,
                            info.numRows));

      // No result from other rank has to be added to first block or for standard reduce case -> start
      // copying to host
      if (i == 0 || accumulateRequired) {
        copy_from_gpu_async(block.blasHandle.stream_handle().get(),
                            GPUArrayConstView1D<T>(block.tileViewGPU.data(),
                                                   info.numCols * info.numRows),
                            HostArrayView1D<T>(block.tileViewHost.data(),
                                               info.numCols * info.numRows));
      }
    }
  }

  state_ = TileState::Prepared;
}

template <typename T> auto RingReduceTileGPU<T>::process_step_ring() -> void {
  const IntType numBlocks = blockInfos_.size();


  assert(ringBlocks_.size() >= 2);
  auto &block = ringBlocks_[currentBlockIdx % ringBlocks_.size()];
  auto &nextBlock = ringBlocks_[(currentBlockIdx + 1) % ringBlocks_.size()];

  if (currentBlockIdx < numBlocks - 1) {
    const auto &recvInfo =
        blockInfos_[(myStartIdx_ + currentBlockIdx + 1) % blockInfos_.size()];
    MPI_Irecv(nextBlock.tileViewHost.data(),
              recvInfo.numCols * recvInfo.numRows,
              MPIMatchElementaryType<T>::get(), recvRank_, ringTag, comm_.get(),
              recvReq_.get_and_activate());

    gpu::check_status(
        gpu::stream_synchronize(block.blasHandle.stream_handle().get()));

    const auto &sendInfo =
        blockInfos_[(myStartIdx_ + currentBlockIdx) % blockInfos_.size()];
    MPI_Send(block.tileViewHost.data(), sendInfo.numRows * sendInfo.numCols,
             MPIMatchElementaryType<T>::get(), sendRank_, ringTag, comm_.get());

    recvReq_.wait_if_active();

    if (nextBlock.matA.rows() != 0) {


      copy_to_gpu_async(
          nextBlock.recvStream.get(),
          HostArrayConstView1D<T>(nextBlock.tileViewHost.data(),
                                  recvInfo.numCols * recvInfo.numRows),
          GPUArrayView1D<T>(nextBlock.recvViewGPU.data(),
                            recvInfo.numCols * recvInfo.numRows));
      nextBlock.event.record(nextBlock.recvStream.get());
      // make sure transfer of received result is done first to avoid scheduling
      // performance issues
      nextBlock.event.stream_wait(nextBlock.blasHandle.stream_handle().get());
      call_gpu_geam(nextBlock.blasHandle.get(), gpu::blas::operation::None,
                    gpu::blas::operation::None, recvInfo.numRows,
                    recvInfo.numCols, RealValueGPU<T>::create(1.0),
                    nextBlock.recvViewGPU.data(), recvInfo.numRows,
                    RealValueGPU<T>::create(1.0), nextBlock.tileViewGPU.data(),
                    recvInfo.numRows, nextBlock.tileViewGPU.data(),
                    recvInfo.numRows);
      copy_from_gpu_async(
          nextBlock.blasHandle.stream_handle().get(),
          GPUArrayConstView1D<T>(nextBlock.tileViewGPU.data(),
                                 recvInfo.numCols * recvInfo.numRows),
          HostArrayView1D<T>(nextBlock.tileViewHost.data(),
                             recvInfo.numCols * recvInfo.numRows));
    }
  }

  if (block.matA.rows() != 0 &&
      currentBlockIdx + ringBlocks_.size() < numBlocks) {
    const auto &info =
        blockInfos_[(myStartIdx_ + currentBlockIdx + ringBlocks_.size()) %
                    blockInfos_.size()];
    const ValueType beta = RealValueGPU<T>::create(0.0);

    auto opAGPU = opA_ == SplaOperation::SPLA_OP_TRANSPOSE
                      ? gpu::blas::operation::Transpose
                      : gpu::blas::operation::ConjugateTranspose;
    // make sure transfer of received result is done first to avoid scheduling
    // performance issues
    nextBlock.event.stream_wait(block.blasHandle.stream_handle().get());
    // compute gemm
    multiply_gpu<ValueType>(
        block.blasHandle.get(), opAGPU, gpu::blas::operation::None, alpha_,
        block.matA.sub_accessor(0, info.globalSubRowIdx, block.matA.rows(),
                                info.numRows),
        block.matB.sub_accessor(0, info.globalSubColIdx, block.matB.rows(),
                                info.numCols),
        beta,
        GPUArrayView2D<T>(block.tileViewGPU.data(), info.numCols,
                          info.numRows));
  }

  if (currentBlockIdx < numBlocks - 1)
    state_ = TileState::PartiallyProcessed;
  else
    state_ = TileState::Processed;
}

template <typename T>
auto RingReduceTileGPU<T>::process_step_reduction() -> void {
  const auto maxBlockSize =
      matrixDist_->max_cols_in_block() * matrixDist_->max_rows_in_block();

  const BlockInfo &info = blockInfos_[currentBlockIdx];
  auto &block = ringBlocks_[currentBlockIdx % ringBlocks_.size()];

  gpu::check_status(
      gpu::stream_synchronize(block.blasHandle.stream_handle().get()));
  mpi_check_status(MPI_Reduce(
      block.tileViewHost.data(),
      resultBufferHost_->data<T>() + numMyBlocksReduced_ * maxBlockSize,
      info.numCols * info.numRows, MPIMatchElementaryType<ValueType>::get(),
      MPI_SUM, info.mpiRank, comm_.get()));

  if (info.mpiRank == comm_.rank())
    ++numMyBlocksReduced_;

  if (block.matA.rows() != 0 && currentBlockIdx + ringBlocks_.size() < blockInfos_.size()) {
    const BlockInfo &recvInfo =
        blockInfos_[currentBlockIdx + ringBlocks_.size()];
    auto opAGPU = opA_ == SplaOperation::SPLA_OP_TRANSPOSE
                      ? gpu::blas::operation::Transpose
                      : gpu::blas::operation::ConjugateTranspose;
    multiply_gpu<ValueType>(
        block.blasHandle.get(), opAGPU, gpu::blas::operation::None, alpha_,
        block.matA.sub_accessor(0, recvInfo.globalSubRowIdx, block.matA.rows(),
                                recvInfo.numRows),
        block.matB.sub_accessor(0, recvInfo.globalSubColIdx, block.matB.rows(),
                                recvInfo.numCols),
        RealValueGPU<T>::create(0.0),
        GPUArrayView2D<T>(block.tileViewGPU.data(), recvInfo.numCols,
                          recvInfo.numRows));
    copy_from_gpu_async(
        block.blasHandle.stream_handle().get(),
        GPUArrayConstView1D<T>(block.tileViewGPU.data(),
                               recvInfo.numCols * recvInfo.numRows),
        HostArrayView1D<T>(block.tileViewHost.data(), recvInfo.numCols * recvInfo.numRows));
  }

  if (currentBlockIdx < blockInfos_.size() - 1)
    state_ = TileState::PartiallyProcessed;
  else
    state_ = TileState::Processed;
}

template <typename T>
auto RingReduceTileGPU<T>::finalize() -> void {
  assert(state_ == TileState::Processed);

  // add tile to result as final step
  const bool accumulateRequired = blockInfos_.size() != comm_.size();
  const auto maxBlockSize =
      matrixDist_->max_cols_in_block() * matrixDist_->max_rows_in_block();
  const bool resultOnHost = GPUMatC_.empty();
  const IntType numBlocks = blockInfos_.size();

  for (auto &b : ringBlocks_) {
    gpu::check_status(
        gpu::stream_synchronize(b.blasHandle.stream_handle().get()));
  }

  if(!accumulateRequired) {
    const auto &info = blockInfos_[(myStartIdx_ + blockInfos_.size() - 1) %
                                   blockInfos_.size()];
    MPI_Send(ringBlocks_[(numBlocks - 1) % ringBlocks_.size()].tileViewHost.data(),
             info.numRows * info.numCols, MPIMatchElementaryType<T>::get(),
             info.mpiRank, resultTag, comm_.get());
  }

  using hostType = typename TypeTranslationHost<T>::type;

  if (resultOnHost) {
    auto matCHostConverted = HostArrayView2D<hostType>(
        reinterpret_cast<hostType *>(HostMatC_.data()), HostMatC_.dim_outer(),
        HostMatC_.dim_inner(), HostMatC_.ld_inner());

    auto betaHost = TypeTranslationHost<T>::convert(this->beta_);
    for (IntType i = 0; i < myBlockIndices_.size(); ++i) {
      resultRecvs_[i].wait_if_active();
      const auto &info = blockInfos_[myBlockIndices_[i]];

      add_kernel(info.numRows, info.numCols,
                 resultBufferHost_->data<hostType>() + i * maxBlockSize,
                 info.numRows, betaHost,
                 &matCHostConverted(info.localColIdx, info.localRowIdx),
                 matCHostConverted.ld_inner());
    }
  } else {
    // result should be placed in gpu memory

    for (IntType i = 0; i < myBlockIndices_.size(); ++i) {
      resultRecvs_[i].wait_if_active();
      const auto &info = blockInfos_[myBlockIndices_[i]];

      copy_to_gpu_async<T, T>(
          ringBlocks_.back().blasHandle.stream_handle().get(),
          HostArrayConstView1D<T>(resultBufferHost_->data<T>() +
                                      i * maxBlockSize,
                                  info.numCols * info.numRows),
          GPUArrayView1D<T>(ringBlocks_.back().tileViewGPU.data(), info.numCols * info.numRows));

      T *subMatPtr =
          GPUMatC_.data() + GPUMatC_.index(info.localColIdx, info.localRowIdx);
      call_gpu_geam(ringBlocks_.back().blasHandle.get(),
                    gpu::blas::operation::None, gpu::blas::operation::None,
                    info.numRows, info.numCols, RealValueGPU<T>::create(1.0),
                    ringBlocks_.back().tileViewGPU.data(), info.numRows, beta_,
                    subMatPtr, GPUMatC_.ld_inner(), subMatPtr,
                    GPUMatC_.ld_inner());
    }
  }

  state_ = TileState::Empty;
}

template <typename T> auto RingReduceTileGPU<T>::process_step() -> bool {
  const bool accumulateRequired = blockInfos_.size() != comm_.size();
  const IntType numBlocks = blockInfos_.size();

  if (currentBlockIdx < numBlocks) {
    if (!accumulateRequired) {
      this->process_step_ring();
    } else {
      this->process_step_reduction();
    }
  } 

  ++currentBlockIdx;
  return currentBlockIdx <= numBlocks;
}

template class RingReduceTileGPU<double>;
template class RingReduceTileGPU<float>;
template class RingReduceTileGPU<gpu::blas::ComplexFloatType>;
template class RingReduceTileGPU<gpu::blas::ComplexDoubleType>;

} // namespace spla
