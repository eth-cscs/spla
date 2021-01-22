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
#include "util/blas_interface.hpp"
#include "util/common_types.hpp"
#include <algorithm>
#include <cassert>
#include <complex>
#include <cstring>
#include <memory>

namespace spla {

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
    std::shared_ptr<Buffer<GPUAllocator>> bufferGPU,
    std::shared_ptr<MatrixBlockGenerator> matrixDist, SplaOperation opA,
    ValueType alpha, GPUMatrixAccessor<GPUArrayConstView2D<ValueType>> matA,
    GPUMatrixAccessor<GPUArrayConstView2D<ValueType>> matB, ValueType beta,
    HostArrayView2D<ValueType> HostMatC, GPUArrayView2D<ValueType> GPUMatC)
    : comm_(std::move(comm)), matrixDist_(std::move(matrixDist)),
      bufferHost_(std::move(bufferHost)), bufferGPU_(std::move(bufferGPU)),
      blasHandle_(std::move(blasHandle)), matA_(matA), matB_(matB),
      HostMatC_(HostMatC), GPUMatC_(GPUMatC), alpha_(alpha), beta_(beta),
      opA_(opA) {

  assert(bufferHost_);
  assert(bufferGPU_);
  assert(opA_ == SplaOperation::SPLA_OP_CONJ_TRANSPOSE ||
         opA_ == SplaOperation::SPLA_OP_TRANSPOSE);
  const auto blockSize =
      matrixDist_->max_cols_in_block() * matrixDist_->max_rows_in_block();
  bufferHost_->resize<ValueType>(3 * blockSize);
  resultView_ = HostArrayView2D<T>(bufferHost_->data<T>(),
                                   matrixDist_->max_cols_in_block(),
                                   matrixDist_->max_rows_in_block());
  sendView_ = HostArrayView2D<T>(bufferHost_->data<T>() + blockSize,
                                 matrixDist_->max_cols_in_block(),
                                 matrixDist_->max_rows_in_block());
  recvView_ = HostArrayView2D<T>(bufferHost_->data<T>() + 2 * blockSize,
                                 matrixDist_->max_cols_in_block(),
                                 matrixDist_->max_rows_in_block());

  bufferGPU_->resize<T>(blockSize);
  tileViewGPU_ =
      GPUArrayView2D<T>(bufferGPU_->data<T>(), matrixDist_->max_cols_in_block(),
                        matrixDist_->max_rows_in_block());

}

template <typename T>
auto RingReduceTileGPU<T>::prepare(IntType blockRowIdx, IntType blockColIdx,
                                   IntType numBlockRows, IntType numBlockCols)
    -> void {

  blockInfos_.resize(numBlockRows * numBlockCols);
  infoForReduce_ = nullptr;

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

  const IntType numBlocks = numBlockRows_ * numBlockCols_;
  const bool accumulateRequired = numBlocks != comm_.size();
  if (accumulateRequired) {
    std::memset(resultView_.data(), 0, resultView_.size() * sizeof(T));
    std::memset(sendView_.data(), 0, sendView_.size() * sizeof(T));
    std::memset(recvView_.data(), 0, recvView_.size() * sizeof(T));
  }
}

template <typename T> auto RingReduceTileGPU<T>::process_step() -> bool {
  const IntType numBlocks = numBlockRows_ * numBlockCols_;
  const bool accumulateRequired = numBlocks != comm_.size();
  const bool resultOnHost = GPUMatC_.empty();

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
      if (currentBlockIdx > 0) {
        gpu::stream_synchronize(blasHandle_.stream_handle().get());
        MPI_Send(sendView_.data(), sendView_.size(),
                 MPIMatchElementaryType<T>::get(), sendRank, 0, comm_.get());
      }
      recvReq_.wait_if_active();
      std::swap(sendView_, recvView_);

      ValueType beta = RealValueGPU<T>::create(0.0);
      if (currentBlockIdx) { // only copy to GPU from second step onwards
        beta = RealValueGPU<T>::create(1.0);
        copy_to_gpu_async<T, T>(blasHandle_.stream_handle().get(),
                                HostArrayView2D<ValueType>(sendView_),
                                tileViewGPU_);
      }
      if (matA_.rows() != 0) {
        auto opAGPU = opA_ == SplaOperation::SPLA_OP_TRANSPOSE
                          ? gpu::blas::operation::Transpose
                          : gpu::blas::operation::ConjugateTranspose;
        multiply_gpu<ValueType>(
            blasHandle_.get(), opAGPU, gpu::blas::operation::None, alpha_,
            matA_.sub_accessor(0, info.globalSubRowIdx,
                               matA_.rows(), info.numRows),
            matB_.sub_accessor(0, info.globalSubColIdx,
                               matB_.rows(), info.numCols),
            beta,
            GPUArrayView2D<T>(tileViewGPU_.data(), info.numCols, info.numRows,
                              tileViewGPU_.ld_inner()));
        if (resultOnHost || (currentBlockIdx < numBlocks - 1)) {
          copy_from_gpu_async(blasHandle_.stream_handle().get(),
                              GPUArrayConstView2D<T>(tileViewGPU_), sendView_);
        }
      }
      if (currentBlockIdx < numBlocks - 1)
        MPI_Irecv(recvView_.data(), recvView_.size(),
                  MPIMatchElementaryType<T>::get(), recvRank, 0, comm_.get(),
                  recvReq_.get_and_activate());
    } else {
      // Number of blocks not equal to number of ranks -> use MPI_Reduce instaed of ring
      assert(accumulateRequired);

      const BlockInfo &info = blockInfos_[currentBlockIdx];

      if(infoForReduce_) {
        sendReq_.wait_if_active();
        gpu::stream_synchronize(blasHandle_.stream_handle().get());
        mpi_check_status(MPI_Ireduce(
            sendView_.data(), resultView_.data(), sendView_.size(),
            MPIMatchElementaryType<ValueType>::get(), MPI_SUM,
            infoForReduce_->mpiRank, comm_.get(), sendReq_.get_and_activate()));
        std::swap(sendView_, recvView_);
      }

      infoForReduce_ = &info;


      if (matA_.rows() != 0) {
        auto opAGPU = opA_ == SplaOperation::SPLA_OP_TRANSPOSE
                          ? gpu::blas::operation::Transpose
                          : gpu::blas::operation::ConjugateTranspose;
        multiply_gpu<ValueType>(
            blasHandle_.get(), opAGPU, gpu::blas::operation::None, alpha_,
            matA_.sub_accessor(0, info.globalSubRowIdx, matA_.rows(),
                               info.numRows),
            matB_.sub_accessor(0, info.globalSubColIdx, matB_.rows(),
                               info.numCols),
            RealValueGPU<T>::create(0.0),
            GPUArrayView2D<T>(tileViewGPU_.data(), info.numCols, info.numRows,
                              tileViewGPU_.ld_inner()));
        copy_from_gpu_async(blasHandle_.stream_handle().get(),
                            GPUArrayConstView2D<T>(tileViewGPU_), sendView_);
      }

    }

  } else if (currentBlockIdx == numBlocks) {
    // add tile to result as final step

    sendReq_.wait_if_active();
    recvReq_.wait_if_active();

    if (accumulateRequired && infoForReduce_) {
      gpu::stream_synchronize(blasHandle_.stream_handle().get());
      mpi_check_status(
          MPI_Reduce(sendView_.data(), resultView_.data(), sendView_.size(),
                     MPIMatchElementaryType<ValueType>::get(), MPI_SUM,
                     infoForReduce_->mpiRank, comm_.get()));
    }

    if (myBlockIdx_ >= 0) {
      auto tileView = accumulateRequired ? resultView_ : sendView_;
      const auto &myInfo = blockInfos_[myBlockIdx_];
      if (resultOnHost) {
        // result should be placed in host memory
        // interpret as std::complex or scalar for host compuations
        auto betaHost = TypeTranslationHost<T>::convert(this->beta_);
        auto matCHostConverted =
            HostArrayView2D<typename TypeTranslationHost<T>::type>(
                reinterpret_cast<typename TypeTranslationHost<T>::type *>(
                    HostMatC_.data()),
                HostMatC_.dim_outer(), HostMatC_.dim_inner(),
                HostMatC_.ld_inner());
        auto tileHostConverted =
            HostArrayView2D<typename TypeTranslationHost<T>::type>(
                reinterpret_cast<typename TypeTranslationHost<T>::type *>(
                    tileView.data()),
                tileView.dim_outer(), tileView.dim_inner(),
                tileView.ld_inner());

        gpu::stream_synchronize(blasHandle_.stream_handle().get());
        if (betaHost == typename TypeTranslationHost<T>::type(0.0)) {
          for (IntType col = 0; col < myInfo.numCols; ++col) {
            for (IntType row = 0; row < myInfo.numRows; ++row) {
              matCHostConverted(myInfo.localColIdx + col,
                                myInfo.localRowIdx + row) =
                  tileHostConverted(col, row);
            }
          }
        } else {
          for (IntType col = 0; col < myInfo.numCols; ++col) {
            for (IntType row = 0; row < myInfo.numRows; ++row) {
              matCHostConverted(myInfo.localColIdx + col,
                                myInfo.localRowIdx + row) =
                  betaHost * matCHostConverted(myInfo.localColIdx + col,
                                               myInfo.localRowIdx + row) +
                  tileHostConverted(col, row);
            }
          }
        }
      } else {
        // result should be placed in gpu memory
        // copy data once if at least one block is assigned to this rank
        if (accumulateRequired) {
          copy_to_gpu_async<ValueType, ValueType>(
              blasHandle_.stream_handle().get(), tileView, tileViewGPU_);
        }
        T *subMatPtr = GPUMatC_.data() +
                       GPUMatC_.index(myInfo.localColIdx, myInfo.localRowIdx);
        call_gpu_geam(blasHandle_.get(), gpu::blas::operation::None,
                      gpu::blas::operation::None, myInfo.numRows,
                      myInfo.numCols, RealValueGPU<T>::create(1.0),
                      tileViewGPU_.data(), tileViewGPU_.ld_inner(), beta_,
                      subMatPtr, GPUMatC_.ld_inner(), subMatPtr,
                      GPUMatC_.ld_inner());
      }
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
