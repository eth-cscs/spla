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
#include "gpu_util/multiply_gpu.hpp"

#include <cmath>
#include <vector>

#include "gpu_util/gpu_blas_api.hpp"
#include "gpu_util/gpu_blas_handle.hpp"
#include "gpu_util/gpu_helper.hpp"
#include "gpu_util/gpu_matrix_accessor.hpp"

namespace spla {

static auto call_gpu_gemm(gpu::blas::HandleType handle, gpu::blas::OperationType transa,
                          gpu::blas::OperationType transb, int m, int n, int k, const float alpha,
                          const float *A, int lda, const float *B, int ldb, const float beta,
                          float *C, int ldc) -> void {
  gpu::blas::check_status(
      gpu::blas::sgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc));
}

static auto call_gpu_gemm(gpu::blas::HandleType handle, gpu::blas::OperationType transa,
                          gpu::blas::OperationType transb, int m, int n, int k, const double alpha,
                          const double *A, int lda, const double *B, int ldb, const double beta,
                          double *C, int ldc) -> void {
  gpu::blas::check_status(
      gpu::blas::dgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc));
}

static auto call_gpu_gemm(gpu::blas::HandleType handle, gpu::blas::OperationType transa,
                          gpu::blas::OperationType transb, int m, int n, int k,
                          const gpu::blas::ComplexFloatType alpha,
                          const gpu::blas::ComplexFloatType *A, int lda,
                          const gpu::blas::ComplexFloatType *B, int ldb,
                          const gpu::blas::ComplexFloatType beta, gpu::blas::ComplexFloatType *C,
                          int ldc) -> void {
  gpu::blas::check_status(
      gpu::blas::cgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc));
}

static auto call_gpu_gemm(gpu::blas::HandleType handle, gpu::blas::OperationType transa,
                          gpu::blas::OperationType transb, int m, int n, int k,
                          const gpu::blas::ComplexDoubleType alpha,
                          const gpu::blas::ComplexDoubleType *A, int lda,
                          const gpu::blas::ComplexDoubleType *B, int ldb,
                          const gpu::blas::ComplexDoubleType beta, gpu::blas::ComplexDoubleType *C,
                          int ldc) -> void {
  gpu::blas::check_status(
      gpu::blas::zgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc));
}

template <typename T>
auto multiply_gpu(const gpu::blas::HandleType &handle, gpu::blas::OperationType transa,
                  gpu::blas::OperationType transb, T alpha,
                  const GPUMatrixAccessor<GPUArrayConstView2D<T>> &tileA,
                  const GPUMatrixAccessor<GPUArrayConstView2D<T>> &tileB, T beta,
                  GPUArrayView2D<T> result) -> void {
  assert(transa == gpu::blas::operation::None || tileA.cols() == result.dim_inner());
  assert(transa != gpu::blas::operation::None || tileA.rows() == result.dim_inner());
  assert(transb == gpu::blas::operation::None || tileB.rows() == result.dim_outer());
  assert(transb != gpu::blas::operation::None || tileB.cols() == result.dim_outer());
  assert(tileA.rows() * tileA.cols() / result.dim_inner() ==
         tileB.rows() * tileB.cols() / result.dim_outer());

  gpu::StreamType stream;
  gpu::blas::get_stream(handle, &stream);

  if (result.size() == 0) return;
  if (tileA.rows() * tileA.cols() == 0) {
    // Scale C only
    call_gpu_gemm(handle, transa, transb, result.dim_inner(), result.dim_outer(), 0, alpha, nullptr,
                  1, nullptr, 1, beta, result.data(), result.ld_inner());
    return;
  }

  IntType innerBlockSize = transa == gpu::blas::operation::None ? tileA.cols() : tileA.rows();
  if (tileA.max_tile_size() < tileA.rows() * tileA.cols()) {
    // if not fully on GPU, try square size
    innerBlockSize = std::min<IntType>(std::sqrt(tileA.max_tile_size()), innerBlockSize);
  }

  if (tileB.max_tile_size() < tileB.rows() * tileB.cols()) {
    // if not fully on GPU, try square size
    innerBlockSize = std::min<IntType>(std::sqrt(tileB.max_tile_size()), innerBlockSize);
  }

  const IntType outerBlockSizeA =
      std::min(tileA.max_tile_size() / innerBlockSize,
               transa == gpu::blas::operation::None ? tileA.rows() : tileA.cols());
  const IntType outerBlockSizeB =
      std::min(tileB.max_tile_size() / innerBlockSize,
               transb == gpu::blas::operation::None ? tileB.cols() : tileB.rows());

  const auto innerSize = transa == gpu::blas::operation::None ? tileA.cols() : tileA.rows();

  for (IntType inner = 0; inner < innerSize; inner += innerBlockSize) {
    const IntType numInner = std::min<IntType>(innerSize - inner, innerBlockSize);

    for (IntType outerA = 0; outerA < result.dim_inner(); outerA += outerBlockSizeA) {
      const IntType numOuterA = std::min<IntType>(result.dim_inner() - outerA, outerBlockSizeA);
      const auto viewA = transa == gpu::blas::operation::None
                             ? tileA.get_tile(outerA, inner, numOuterA, numInner, stream)
                             : tileA.get_tile(inner, outerA, numInner, numOuterA, stream);

      for (IntType outerB = 0; outerB < result.dim_outer(); outerB += outerBlockSizeB) {
        const IntType numOuterB = std::min<IntType>(result.dim_outer() - outerB, outerBlockSizeB);
        const auto viewB = transb == gpu::blas::operation::None
                               ? tileB.get_tile(inner, outerB, numInner, numOuterB, stream)
                               : tileB.get_tile(outerB, inner, numOuterB, numInner, stream);

        call_gpu_gemm(handle, transa, transb, numOuterA, numOuterB, numInner, alpha, viewA.data(),
                      viewA.ld_inner(), viewB.data(), viewB.ld_inner(), beta,
                      result.data() + result.index(outerB, outerA), result.ld_inner());
      }
    }
    beta = RealValueGPU<T>::create(1.0);
  }
}

template auto multiply_gpu<float>(const gpu::blas::HandleType &handle,
                                  gpu::blas::OperationType transa, gpu::blas::OperationType transb,
                                  float alpha,
                                  const GPUMatrixAccessor<GPUArrayConstView2D<float>> &tileA,
                                  const GPUMatrixAccessor<GPUArrayConstView2D<float>> &tileB,
                                  float beta, GPUArrayView2D<float> result) -> void;

template auto multiply_gpu<double>(const gpu::blas::HandleType &handle,
                                   gpu::blas::OperationType transa, gpu::blas::OperationType transb,
                                   double alpha,
                                   const GPUMatrixAccessor<GPUArrayConstView2D<double>> &tileA,
                                   const GPUMatrixAccessor<GPUArrayConstView2D<double>> &tileB,
                                   double beta, GPUArrayView2D<double> result) -> void;

template auto multiply_gpu<gpu::blas::ComplexFloatType>(
    const gpu::blas::HandleType &handle, gpu::blas::OperationType transa,
    gpu::blas::OperationType transb, gpu::blas::ComplexFloatType alpha,
    const GPUMatrixAccessor<GPUArrayConstView2D<gpu::blas::ComplexFloatType>> &tileA,
    const GPUMatrixAccessor<GPUArrayConstView2D<gpu::blas::ComplexFloatType>> &tileB,
    gpu::blas::ComplexFloatType beta, GPUArrayView2D<gpu::blas::ComplexFloatType> result) -> void;

template auto multiply_gpu<gpu::blas::ComplexDoubleType>(
    const gpu::blas::HandleType &handle, gpu::blas::OperationType transa,
    gpu::blas::OperationType transb, gpu::blas::ComplexDoubleType alpha,
    const GPUMatrixAccessor<GPUArrayConstView2D<gpu::blas::ComplexDoubleType>> &tileA,
    const GPUMatrixAccessor<GPUArrayConstView2D<gpu::blas::ComplexDoubleType>> &tileB,
    gpu::blas::ComplexDoubleType beta, GPUArrayView2D<gpu::blas::ComplexDoubleType> result) -> void;

}  // namespace spla
