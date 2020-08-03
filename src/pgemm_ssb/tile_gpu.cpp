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
#include <algorithm>
#include <cassert>
#include <complex>
#include <cassert>
#include <cstring>
#include <memory>
#include "gpu_util/gpu_runtime_api.hpp"
#include "gpu_util/gpu_blas_api.hpp"
#include "memory/gpu_array_const_view.hpp"
#include "memory/host_array_view.hpp"
#include "mpi_util/mpi_check_status.hpp"
#include "mpi_util/mpi_match_elementary_type.hpp"
#include "gpu_util/gpu_transfer.hpp"
#include "gpu_util/gpu_helper.hpp"
#include "util/blas_interface.hpp"
#include "util/common_types.hpp"
#include "gpu_util/multiply_gpu.hpp"
#include "pgemm_ssb/tile_gpu.hpp"
#include "block_generation/matrix_block_generator.hpp"

namespace spla {

static auto call_gpu_geam(const gpu::blas::HandleType& handle,
                          const gpu::blas::OperationType& transa,
                          const gpu::blas::OperationType& transb, int m, int n, float alpha,
                          const float* A, int lda, float beta, const float* B, int ldb, float* C,
                          int ldc) -> void {
  gpu::blas::sgeam(handle, transa, transb, m, n, &alpha, A, lda, &beta, B, ldb, C, ldc);
}

static auto call_gpu_geam(const gpu::blas::HandleType& handle,
                          const gpu::blas::OperationType& transa,
                          const gpu::blas::OperationType& transb, int m, int n, double alpha,
                          const double* A, int lda, double beta, const double* B, int ldb,
                          double* C, int ldc) -> void {
  gpu::blas::dgeam(handle, transa, transb, m, n, &alpha, A, lda, &beta, B, ldb, C, ldc);
}

static auto call_gpu_geam(const gpu::blas::HandleType& handle,
                          const gpu::blas::OperationType& transa,
                          const gpu::blas::OperationType& transb, int m, int n,
                          gpu::blas::ComplexDoubleType alpha, const gpu::blas::ComplexDoubleType* A,
                          int lda, gpu::blas::ComplexDoubleType beta,
                          const gpu::blas::ComplexDoubleType* B, int ldb,
                          gpu::blas::ComplexDoubleType* C, int ldc) -> void {
  gpu::blas::zgeam(handle, transa, transb, m, n, &alpha, A, lda, &beta, B, ldb, C, ldc);
}

static auto call_gpu_geam(const gpu::blas::HandleType& handle,
                          const gpu::blas::OperationType& transa,
                          const gpu::blas::OperationType& transb, int m, int n,
                          gpu::blas::ComplexFloatType alpha, const gpu::blas::ComplexFloatType* A,
                          int lda, gpu::blas::ComplexFloatType beta,
                          const gpu::blas::ComplexFloatType* B, int ldb,
                          gpu::blas::ComplexFloatType* C, int ldc) -> void {
  gpu::blas::cgeam(handle, transa, transb, m, n, &alpha, A, lda, &beta, B, ldb, C, ldc);
}
template <typename T>
TileGPU<T>::TileGPU(MPICommunicatorHandle comm, GPUBlasHandle blasHandle,
                    std::shared_ptr<Buffer<PinnedAllocator>> bufferHost,
                    std::shared_ptr<Buffer<GPUAllocator>> bufferGPU,
                    std::shared_ptr<MatrixBlockGenerator> matrixDist, SplaOperation opA,
                    ValueType alpha, GPUMatrixAccessor<GPUArrayConstView2D<ValueType>> matA,
                    GPUMatrixAccessor<GPUArrayConstView2D<ValueType>> matB, ValueType beta,
                    HostArrayView2D<ValueType> HostMatC, GPUArrayView2D<ValueType> GPUMatC,
                    IntType numBlockRows, IntType numBlockCols)
    : state_(TileState::Empty),
      comm_(std::move(comm)),
      matrixDist_(std::move(matrixDist)),
      bufferHost_(std::move(bufferHost)),
      bufferGPU_(std::move(bufferGPU)),
      blasHandle_(std::move(blasHandle)),
      numBlockRows_(numBlockRows),
      numBlockCols_(numBlockCols),
      matA_(matA),
      matB_(matB),
      HostMatC_(HostMatC),
      GPUMatC_(GPUMatC),
      alpha_(alpha),
      beta_(beta),
      opA_(opA) {
  assert(matA_.rows() == matB_.rows());
  assert(bufferHost_);
  assert(bufferGPU_);
  assert(opA_ == SplaOperation::SPLA_OP_CONJ_TRANSPOSE || opA_ == SplaOperation::SPLA_OP_TRANSPOSE);
  const auto bufferSize = numBlockRows * numBlockCols * matrixDist_->max_rows_in_block() * matrixDist_->max_cols_in_block();
  bufferHost_->resize<ValueType>(bufferSize);
  bufferGPU_->resize<ValueType>(bufferSize);
}

template <typename T>
auto TileGPU<T>::multiply(IntType blockRowIdx, IntType blockColIdx) -> void {
  assert(state_.get() == TileState::Empty);
  if (state_.get() != TileState::Empty) {
    throw InternalError();
  }

  const IntType numCurrentBlockRows = std::min(matrixDist_->num_block_rows() - blockRowIdx, numBlockRows_);
  const IntType numCurrentBlockCols = std::min(matrixDist_->num_block_cols() - blockColIdx, numBlockCols_);

  // get block informations
  blockInfos_.clear(); // leaves capacity unchanged
  blockInfos_.reserve(numCurrentBlockRows * numCurrentBlockCols);
  for (IntType c = 0; c < numCurrentBlockCols; ++c) {
    for (IntType r = 0; r < numCurrentBlockRows; ++r) {
      blockInfos_.emplace_back(matrixDist_->get_block_info(blockRowIdx + r, blockColIdx +c));
    }
  }

  const IntType numTileRows = blockInfos_.back().numRows + blockInfos_.back().globalSubRowIdx -
                              blockInfos_.front().globalSubRowIdx;
  const IntType numTileCols = blockInfos_.back().numCols + blockInfos_.back().globalSubColIdx -
                              blockInfos_.front().globalSubColIdx;

  tileHost_ =
      HostArrayView2D<ValueType>(bufferHost_->data<ValueType>(), numTileCols, numTileRows);
  tileGPU_ = GPUArrayView2D<ValueType>(bufferGPU_->data<ValueType>(), numTileCols, numTileRows);

  if (matA_.rows() == 0) {
    std::memset(tileHost_.data(), 0, tileHost_.size() * sizeof(T));
  } else {
    ValueType beta = RealValueGPU<T>::create(0.0);
    auto opAGPU = opA_ == SplaOperation::SPLA_OP_TRANSPOSE
                      ? gpu::blas::operation::Transpose
                      : gpu::blas::operation::ConjugateTranspose;
    multiply_gpu<ValueType>(
        blasHandle_.get(), opAGPU, gpu::blas::operation::None, alpha_,
        matA_.sub_accessor(0, blockInfos_.front().globalSubRowIdx, matA_.rows(), numTileRows),
        matB_.sub_accessor(0, blockInfos_.front().globalSubColIdx, matB_.rows(), numTileCols), beta,
        tileGPU_);
    copy_from_gpu_async(blasHandle_.stream_handle().get(), GPUArrayConstView2D<T>(tileGPU_),
                        tileHost_);
  }

  // set state atomically
  state_.set(TileState::Multiplied);
}

template <typename T>
auto TileGPU<T>::exchange() -> void {
  assert(this->state_.get() == TileState::Multiplied);
  if (this->state_.get() != TileState::Multiplied) {
    throw InternalError();
  }

  // wait for GPU to finish multiplication
  gpu::check_status(gpu::stream_synchronize(blasHandle_.stream_handle().get()));

  if(blockInfos_.size() == 1 && blockInfos_.front().mpiRank >= 0) {
    const auto& info = blockInfos_.front();
    // result is send to single rank
    if (comm_.rank() == info.mpiRank) {
      mpi_check_status(MPI_Reduce(MPI_IN_PLACE, this->tileHost_.data(), this->tileHost_.size(),
                                   MPIMatchElementaryType<ValueType>::get(), MPI_SUM, info.mpiRank,
                                   comm_.get()));
    } else {
      mpi_check_status(MPI_Reduce(this->tileHost_.data(), nullptr, this->tileHost_.size(),
                                   MPIMatchElementaryType<ValueType>::get(), MPI_SUM, info.mpiRank,
                                   comm_.get()));
    }
  } else {
    // result required on all ranks
    mpi_check_status(MPI_Allreduce(MPI_IN_PLACE, this->tileHost_.data(), this->tileHost_.size(),
                                    MPIMatchElementaryType<ValueType>::get(), MPI_SUM, comm_.get()));
  }

  // set state atomically
  this->state_.set(TileState::Exchanged);
}

template <typename T>
auto TileGPU<T>::extract() -> void {
  assert(this->state_.get() == TileState::Exchanged);
  if (this->state_.get() != TileState::Exchanged) {
    throw InternalError();
  }

  if (GPUMatC_.empty()) {
    // result should be placed in host memory
    for (const auto& info : blockInfos_) {
      const IntType tileRowOffset = info.globalSubRowIdx - blockInfos_.front().globalSubRowIdx;
      const IntType tileColOffset = info.globalSubColIdx - blockInfos_.front().globalSubColIdx;
      if (info.mpiRank == comm_.rank() || info.mpiRank < 0) {
        // interpret as std::complex or scalar for host compuations
        auto betaHost = TypeTranslationHost<T>::convert(this->beta_);
        auto matCHostConverted = HostArrayView2D<typename TypeTranslationHost<T>::type>(
            reinterpret_cast<typename TypeTranslationHost<T>::type*>(HostMatC_.data()),
            HostMatC_.dim_outer(), HostMatC_.dim_inner(), HostMatC_.ld_inner());
        auto tileHostConverted = HostArrayView2D<typename TypeTranslationHost<T>::type>(
            reinterpret_cast<typename TypeTranslationHost<T>::type*>(tileHost_.data()),
            tileHost_.dim_outer(), tileHost_.dim_inner(), tileHost_.ld_inner());

        if (betaHost == typename TypeTranslationHost<T>::type(0.0)) {
          for (IntType col = 0; col < info.numCols; ++col) {
            for (IntType row = 0; row < info.numRows; ++row) {
              matCHostConverted(info.localColIdx + col, info.localRowIdx + row) =
                  tileHostConverted(col + tileColOffset, row + tileRowOffset);
            }
          }
        } else {
          for (IntType col = 0; col < info.numCols; ++col) {
            for (IntType row = 0; row < info.numRows; ++row) {
              matCHostConverted(info.localColIdx + col, info.localRowIdx + row) =
                  betaHost * matCHostConverted(info.localColIdx + col, info.localRowIdx + row) +
                  tileHostConverted(col + tileColOffset, row + tileRowOffset);
            }
          }
        }
      }
    }
  } else {
    // result should be placed in gpu memory
    copy_to_gpu_async<ValueType, ValueType>(blasHandle_.stream_handle().get(), tileHost_, tileGPU_);
    for (const auto& info : blockInfos_) {
      const IntType tileRowOffset = info.globalSubRowIdx - blockInfos_.front().globalSubRowIdx;
      const IntType tileColOffset = info.globalSubColIdx - blockInfos_.front().globalSubColIdx;
      T* subTilePtr = tileGPU_.data() + tileGPU_.index(tileColOffset, tileRowOffset);
      T* subMatPtr = GPUMatC_.data() + GPUMatC_.index(info.localColIdx, info.localRowIdx);
      call_gpu_geam(blasHandle_.get(), gpu::blas::operation::None, gpu::blas::operation::None,
                    info.numRows, info.numCols, RealValueGPU<T>::create(1.0),
                    subTilePtr, tileGPU_.ld_inner(), beta_, subMatPtr, GPUMatC_.ld_inner(),
                    subMatPtr, GPUMatC_.ld_inner());
    }
  }

  // set state atomically
  this->state_.set(TileState::Empty);
}

template class TileGPU<double>;
template class TileGPU<float>;
template class TileGPU<gpu::blas::ComplexFloatType>;
template class TileGPU<gpu::blas::ComplexDoubleType>;

}  // namespace spla
