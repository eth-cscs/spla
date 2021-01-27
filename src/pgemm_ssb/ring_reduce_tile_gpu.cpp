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
    MPICommunicatorHandle comm, GPUBlasHandle blasHandle,
    std::shared_ptr<Buffer<PinnedAllocator>> bufferHost,
    std::shared_ptr<Buffer<PinnedAllocator>> resultBufferHost,
    std::shared_ptr<Buffer<GPUAllocator>> bufferGPU,
    std::shared_ptr<MatrixBlockGenerator> matrixDist, SplaOperation opA,
    ValueType alpha, GPUMatrixAccessor<GPUArrayConstView2D<ValueType>> matA,
    GPUMatrixAccessor<GPUArrayConstView2D<ValueType>> matB, ValueType beta,
    HostArrayView2D<ValueType> HostMatC, GPUArrayView2D<ValueType> GPUMatC)
    : state_(TileState::Empty), comm_(std::move(comm)),
      matrixDist_(std::move(matrixDist)), bufferHost_(std::move(bufferHost)),
      resultBufferHost_(std::move(resultBufferHost)),
      bufferGPU_(std::move(bufferGPU)), blasHandle_(std::move(blasHandle)),
      matA_(matA), matB_(matB), HostMatC_(HostMatC), GPUMatC_(GPUMatC),
      alpha_(alpha), beta_(beta), opA_(opA) {

  assert(bufferHost_);
  assert(resultBufferHost_);
  assert(bufferGPU_);
  assert(opA_ == SplaOperation::SPLA_OP_CONJ_TRANSPOSE ||
         opA_ == SplaOperation::SPLA_OP_TRANSPOSE);
  const auto blockSize =
      matrixDist_->max_cols_in_block() * matrixDist_->max_rows_in_block();
  bufferHost_->resize<ValueType>(3 * blockSize);
  sendView_ = HostArrayView1D<T>(bufferHost_->data<T>(), blockSize);
  recvView_ = HostArrayView1D<T>(bufferHost_->data<T>() + blockSize, blockSize);
  processingView_ =
      HostArrayView1D<T>(bufferHost_->data<T>() + 2 * blockSize, blockSize);

  bufferGPU_->resize<T>(blockSize);
  tileViewGPU_ = GPUArrayView1D<T>(bufferGPU_->data<T>(), blockSize);
}

template <typename T>
auto RingReduceTileGPU<T>::prepare(std::vector<BlockInfo>::const_iterator begin,
                                   std::vector<BlockInfo>::const_iterator end)
    -> void {
  assert(state_ == TileState::Empty);
  gpu::stream_synchronize(blasHandle_.stream_handle().get());

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

  resultBufferHost_->resize<T>(
      std::max<std::size_t>(myBlockIndices_.size(), 1) * maxBlockSize);
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

  state_ = TileState::Prepared;
}

template <typename T> auto RingReduceTileGPU<T>::process_step_ring() -> void {
  const IntType numBlocks = blockInfos_.size();

  const auto &info =
      blockInfos_[(myStartIdx_ + currentBlockIdx) % blockInfos_.size()];
  assert(info.mpiRank >= 0); // Mirror distribution not supported
  const auto &nextInfo =
      blockInfos_[(myStartIdx_ + currentBlockIdx + 1) % blockInfos_.size()];

  gpu::stream_synchronize(blasHandle_.stream_handle().get());

  std::swap(processingView_, recvView_);
  std::swap(recvView_, sendView_);

  sendReq_.wait_if_active();
  if (currentBlockIdx > 0) {
    const auto previousInfo =
        blockInfos_[(myStartIdx_ + currentBlockIdx - 1) % blockInfos_.size()];
    MPI_Isend(sendView_.data(), previousInfo.numRows * previousInfo.numCols,
              MPIMatchElementaryType<T>::get(), sendRank_, ringTag, comm_.get(),
              sendReq_.get_and_activate());
  }

  recvReq_.wait_if_active();
  if (currentBlockIdx < numBlocks - 1) {
    MPI_Irecv(recvView_.data(), nextInfo.numCols * nextInfo.numRows,
              MPIMatchElementaryType<T>::get(), recvRank_, ringTag, comm_.get(),
              recvReq_.get_and_activate());
  }

  if (matA_.rows() != 0) {
    ValueType beta = RealValueGPU<T>::create(0.0);
    if (currentBlockIdx) { // only copy to GPU from second step onwards
      beta = RealValueGPU<T>::create(1.0);
      copy_to_gpu_async<T, T>(
          blasHandle_.stream_handle().get(),
          HostArrayConstView1D<T>(processingView_.data(),
                                  info.numCols * info.numRows),
          GPUArrayView1D<T>(tileViewGPU_.data(), info.numCols * info.numRows));
    }

    auto opAGPU = opA_ == SplaOperation::SPLA_OP_TRANSPOSE
                      ? gpu::blas::operation::Transpose
                      : gpu::blas::operation::ConjugateTranspose;
    multiply_gpu<ValueType>(
        blasHandle_.get(), opAGPU, gpu::blas::operation::None, alpha_,
        matA_.sub_accessor(0, info.globalSubRowIdx, matA_.rows(), info.numRows),
        matB_.sub_accessor(0, info.globalSubColIdx, matB_.rows(), info.numCols),
        beta,
        GPUArrayView2D<T>(tileViewGPU_.data(), info.numCols, info.numRows));
    copy_from_gpu_async(blasHandle_.stream_handle().get(),
                        GPUArrayConstView1D<T>(tileViewGPU_.data(),
                                               info.numCols * info.numRows),
                        HostArrayView1D<T>(processingView_.data(),
                                           info.numCols * info.numRows));
  }

  if (currentBlockIdx == numBlocks - 1) {
    // send final result to target rank
    sendReq_.wait_if_active();
    gpu::stream_synchronize(blasHandle_.stream_handle().get());
    MPI_Isend(processingView_.data(), info.numRows * info.numCols,
              MPIMatchElementaryType<T>::get(), info.mpiRank, resultTag,
              comm_.get(), sendReq_.get_and_activate());
  }

  state_ = TileState::PartiallyProcessed;
}

template <typename T>
auto RingReduceTileGPU<T>::process_step_reduction() -> void {
  const auto maxBlockSize =
      matrixDist_->max_cols_in_block() * matrixDist_->max_rows_in_block();

  const BlockInfo &info = blockInfos_[currentBlockIdx];

  if (currentBlockIdx > 0) {
    const auto previousInfo = blockInfos_[(currentBlockIdx - 1)];
    sendReq_.wait_if_active();
    gpu::stream_synchronize(blasHandle_.stream_handle().get());
    mpi_check_status(MPI_Ireduce(
        sendView_.data(),
        resultBufferHost_->data<T>() + numMyBlocksReduced_ * maxBlockSize,
        previousInfo.numCols * previousInfo.numRows,
        MPIMatchElementaryType<ValueType>::get(), MPI_SUM, previousInfo.mpiRank,
        comm_.get(), sendReq_.get_and_activate()));

    if (previousInfo.mpiRank == comm_.rank())
      ++numMyBlocksReduced_;

    std::swap(recvView_, sendView_);
  }

  if (matA_.rows() != 0) {
    auto opAGPU = opA_ == SplaOperation::SPLA_OP_TRANSPOSE
                      ? gpu::blas::operation::Transpose
                      : gpu::blas::operation::ConjugateTranspose;
    multiply_gpu<ValueType>(
        blasHandle_.get(), opAGPU, gpu::blas::operation::None, alpha_,
        matA_.sub_accessor(0, info.globalSubRowIdx, matA_.rows(), info.numRows),
        matB_.sub_accessor(0, info.globalSubColIdx, matB_.rows(), info.numCols),
        RealValueGPU<T>::create(0.0),
        GPUArrayView2D<T>(tileViewGPU_.data(), info.numCols, info.numRows));
    copy_from_gpu_async(
        blasHandle_.stream_handle().get(),
        GPUArrayConstView1D<T>(tileViewGPU_.data(),
                               info.numCols * info.numRows),
        HostArrayView1D<T>(sendView_.data(), info.numCols * info.numRows));
  } else {
    std::memset(sendView_.data(), 0, info.numCols * info.numRows * sizeof(T));
  }

  if (currentBlockIdx == blockInfos_.size() - 1) {
    sendReq_.wait_if_active();
    gpu::stream_synchronize(blasHandle_.stream_handle().get());
    mpi_check_status(MPI_Ireduce(
        sendView_.data(),
        resultBufferHost_->data<T>() + numMyBlocksReduced_ * maxBlockSize,
        info.numCols * info.numRows, MPIMatchElementaryType<ValueType>::get(),
        MPI_SUM, info.mpiRank, comm_.get(), sendReq_.get_and_activate()));
  }

  state_ = TileState::PartiallyProcessed;
}

template <typename T>
auto RingReduceTileGPU<T>::process_step_finalize() -> void {
  // add tile to result as final step

  sendReq_.wait_if_active();
  recvReq_.wait_if_active();

  const bool resultOnHost = GPUMatC_.empty();
  const auto maxBlockSize =
      matrixDist_->max_cols_in_block() * matrixDist_->max_rows_in_block();

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
          blasHandle_.stream_handle().get(),
          HostArrayConstView1D<T>(resultBufferHost_->data<T>() +
                                      i * maxBlockSize,
                                  info.numCols * info.numRows),
          GPUArrayView1D<T>(tileViewGPU_.data(), info.numCols * info.numRows));

      T *subMatPtr =
          GPUMatC_.data() + GPUMatC_.index(info.localColIdx, info.localRowIdx);
      call_gpu_geam(blasHandle_.get(), gpu::blas::operation::None,
                    gpu::blas::operation::None, info.numRows, info.numCols,
                    RealValueGPU<T>::create(1.0), tileViewGPU_.data(),
                    info.numRows, beta_, subMatPtr, GPUMatC_.ld_inner(),
                    subMatPtr, GPUMatC_.ld_inner());
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
  } else if (currentBlockIdx == numBlocks && currentBlockIdx > 0) {
    this->process_step_finalize();
  }

  ++currentBlockIdx;
  return currentBlockIdx <= numBlocks;
}

template class RingReduceTileGPU<double>;
template class RingReduceTileGPU<float>;
template class RingReduceTileGPU<gpu::blas::ComplexFloatType>;
template class RingReduceTileGPU<gpu::blas::ComplexDoubleType>;

} // namespace spla
