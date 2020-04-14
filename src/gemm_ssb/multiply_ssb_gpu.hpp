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
#ifndef SPLA_MULTIPLY_SSB_GPU_HPP
#define SPLA_MULTIPLY_SSB_GPU_HPP

#include <cmath>
#include "gpu_util/gpu_blas_handle.hpp"
#include "gpu_util/gpu_matrix_accessor.hpp"
#include "gpu_util/gpu_helper.hpp"

namespace spla {

inline auto call_gpu_gemm(gpu::blas::HandleType handle, gpu::blas::OperationType transa,
                          gpu::blas::OperationType transb, int m, int n, int k, const float alpha,
                          const float *A, int lda, const float *B, int ldb, const float beta,
                          float *C, int ldc) -> void {
  gpu::blas::check_status(
      gpu::blas::sgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc));
}

inline auto call_gpu_gemm(gpu::blas::HandleType handle, gpu::blas::OperationType transa,
                          gpu::blas::OperationType transb, int m, int n, int k, const double alpha,
                          const double *A, int lda, const double *B, int ldb, const double beta,
                          double *C, int ldc) -> void {
  gpu::blas::check_status(
      gpu::blas::dgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc));
}

inline auto call_gpu_gemm(gpu::blas::HandleType handle, gpu::blas::OperationType transa,
                          gpu::blas::OperationType transb, int m, int n, int k,
                          const gpu::blas::ComplexFloatType alpha,
                          const gpu::blas::ComplexFloatType *A, int lda,
                          const gpu::blas::ComplexFloatType *B, int ldb,
                          const gpu::blas::ComplexFloatType beta, gpu::blas::ComplexFloatType *C,
                          int ldc) -> void {
  gpu::blas::check_status(
      gpu::blas::cgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc));
}

inline auto call_gpu_gemm(gpu::blas::HandleType handle, gpu::blas::OperationType transa,
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
auto multiply_ssb_gpu(const gpu::blas::HandleType &handle, T alpha,
                      const GPUMatrixAccessor<GPUArrayConstView2D<T>> &tileA,
                      const GPUMatrixAccessor<GPUArrayConstView2D<T>> &tileB, T beta,
                      GPUArrayView2D<T> result) -> void {
  assert(tileA.cols() <= result.dim_inner());
  assert(tileB.cols() <= result.dim_outer());
  assert(tileA.rows() == tileB.rows());

  gpu::StreamType stream;
  gpu::blas::get_stream(handle, &stream);

  IntType rowBlockSize = tileA.rows();
  if(tileA.max_tile_size() < tileA.rows() * tileA.cols()) {
    // if not fully on GPU, try square size
    rowBlockSize = std::min<IntType>(std::sqrt(tileA.max_tile_size()), rowBlockSize);
  }

  if(tileB.max_tile_size() < tileB.rows() * tileB.cols()) {
    // if not fully on GPU, try square size
    rowBlockSize = std::min<IntType>(std::sqrt(tileB.max_tile_size()), rowBlockSize);
  }

  const IntType colBlockSizeA = std::min(tileA.max_tile_size() / rowBlockSize, tileA.cols());
  const IntType colBlockSizeB = std::min(tileB.max_tile_size() / rowBlockSize, tileB.cols());

  for (IntType row = 0; row < tileA.rows(); row += rowBlockSize) {
    const IntType numRows = std::min<IntType>(tileA.rows() - row, rowBlockSize);

    for(IntType colA = 0; colA < tileA.cols(); colA += colBlockSizeA) {
      const IntType numColsA = std::min<IntType>(tileA.cols() - colA, colBlockSizeA);
      const auto viewA = tileA.get_tile(row, colA, numRows, numColsA, stream);

      for (IntType colB = 0; colB < tileB.cols(); colB += colBlockSizeB) {
        const IntType numColsB = std::min<IntType>(tileB.cols() - colB, colBlockSizeB);
        const auto viewB = tileB.get_tile(row, colB, numRows, numColsB, stream);

        call_gpu_gemm(handle, gpu::blas::operation::ConjugateTranspose, gpu::blas::operation::None,
                      numColsA, numColsB, numRows, alpha, viewA.data(), viewA.ld_inner(),
                      viewB.data(), viewB.ld_inner(), beta,
                      result.data() + result.index(colB, colA), result.ld_inner());
      }

    }
    beta = RealValueGPU<T>::create(1.0);
  }
}

}  // namespace spla
#endif
