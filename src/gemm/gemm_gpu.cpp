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

#include "gemm/gemm_gpu.hpp"

#include <cmath>
#include <vector>

#include "gemm/gemm_host.hpp"
#include "gpu_util/gpu_blas_api.hpp"
#include "gpu_util/gpu_blas_handle.hpp"
#include "gpu_util/gpu_complex_type_conversion.hpp"
#include "gpu_util/gpu_device_guard.hpp"
#include "gpu_util/gpu_matrix_accessor.hpp"
#include "gpu_util/gpu_pointer_translation.hpp"
#include "gpu_util/multiply_gpu.hpp"
#include "memory/gpu_array_const_view.hpp"
#include "memory/gpu_array_view.hpp"
#include "util/check_gemm_param.hpp"

namespace spla {

static auto map_op_to_gpu_blas(SplaOperation op) -> gpu::blas::OperationType {
  switch (op) {
    case SplaOperation::SPLA_OP_TRANSPOSE:
      return gpu::blas::operation::Transpose;
    case SplaOperation::SPLA_OP_CONJ_TRANSPOSE:
      return gpu::blas::operation::ConjugateTranspose;
    default:
      return gpu::blas::operation::None;
  }
}

template <typename T>
void gemm_gpu(SplaOperation opA, SplaOperation opB, IntType m, IntType n, IntType k, T alpha,
              const T *A, IntType lda, const T *B, IntType ldb, T beta, T *C, IntType ldc,
              ContextInternal &ctx) {
  if (m == 0 || n == 0) {
    return;
  }

  check_gemm_param(opA, opB, m, n, k, A, lda, B, ldb, C, ldc);

  GPUDeviceGuard deviceGuard(ctx.gpu_device_id());

  // always synchronize with stream 0 as part of API requirement
  gpu::check_status(gpu::stream_synchronize(nullptr));

  const IntType numColsA = opA == SplaOperation::SPLA_OP_NONE ? k : m;
  const IntType numRowsA = opA == SplaOperation::SPLA_OP_NONE ? m : k;
  const IntType numColsB = opB == SplaOperation::SPLA_OP_NONE ? n : k;
  const IntType numRowsB = opB == SplaOperation::SPLA_OP_NONE ? k : n;

  const auto opBlasA = map_op_to_gpu_blas(opA);
  const auto opBlasB = map_op_to_gpu_blas(opB);

  const T *hostPtrA;
  const T *gpuPtrA;
  const T *hostPtrB;
  const T *gpuPtrB;
  T *hostPtrC;
  T *gpuPtrC;

  std::tie(hostPtrA, gpuPtrA) = translate_gpu_pointer(A);
  std::tie(hostPtrB, gpuPtrB) = translate_gpu_pointer(B);
  std::tie(hostPtrC, gpuPtrC) = translate_gpu_pointer(C);

  // Compute on Host if below threshold and input / output not on GPU
  if (!gpuPtrA && !gpuPtrB && !gpuPtrC &&
      k * n < ctx.op_threshold_gpu() / (2 * m)) {  // m always != 0 here
    using hostType = typename ComplexTypeHost<T>::type;
    return gemm_host<hostType>(
        ctx.num_threads(), opA, opB, m, n, k, *reinterpret_cast<hostType *>(&alpha),
        reinterpret_cast<const hostType *>(A), lda, reinterpret_cast<const hostType *>(B), ldb,
        *reinterpret_cast<hostType *>(&beta), reinterpret_cast<hostType *>(C), ldc);
  }

  auto &blasHandles = ctx.gpu_blas_handles(ctx.num_tiles());
  auto &gpuBuffers = ctx.gpu_buffers(3 * ctx.num_tiles());
  std::vector<GPUMatrixAccessor<GPUArrayConstView2D<T>>> matAccessorsA;
  std::vector<GPUMatrixAccessor<GPUArrayConstView2D<T>>> matAccessorsB;
  std::vector<GPUMatrixAccessor<GPUArrayView2D<T>>> matAccessorsC;

  const IntType maxNumElementsInTile = ctx.tile_size_gpu() * ctx.tile_size_gpu();

  for (IntType i = 0; i < ctx.num_tiles(); ++i) {
    matAccessorsA.emplace_back(gpuPtrA
                                   ? GPUMatrixAccessor<GPUArrayConstView2D<T>>(
                                         GPUArrayConstView2D<T>(gpuPtrA, numColsA, numRowsA, lda))
                                   : GPUMatrixAccessor<GPUArrayConstView2D<T>>(
                                         HostArrayConstView2D<T>(A, numColsA, numRowsA, lda),
                                         maxNumElementsInTile, gpuBuffers[i * 3]));

    matAccessorsB.emplace_back(gpuPtrB
                                   ? GPUMatrixAccessor<GPUArrayConstView2D<T>>(
                                         GPUArrayConstView2D<T>(gpuPtrB, numColsB, numRowsB, ldb))
                                   : GPUMatrixAccessor<GPUArrayConstView2D<T>>(
                                         HostArrayConstView2D<T>(B, numColsB, numRowsB, ldb),
                                         maxNumElementsInTile, gpuBuffers[i * 3 + 1]));

    matAccessorsC.emplace_back(
        gpuPtrC
            ? GPUMatrixAccessor<GPUArrayView2D<T>>(GPUArrayView2D<T>(gpuPtrC, n, m, ldc))
            : GPUMatrixAccessor<GPUArrayView2D<T>>(HostArrayConstView2D<T>(C, n, m, ldc),
                                                   maxNumElementsInTile, gpuBuffers[i * 3 + 2]));
  }

  IntType rowBlockSize = m;
  if (matAccessorsC.front().max_tile_size() < n * m) {
    // if not fully on GPU, try square size
    rowBlockSize =
        std::min<IntType>(std::sqrt(matAccessorsC.front().max_tile_size()), rowBlockSize);
  }

  const IntType colBlockSize = std::min(matAccessorsC.front().max_tile_size() / rowBlockSize, n);
  rowBlockSize = std::min(matAccessorsC.front().max_tile_size() / colBlockSize, m);

  IntType counter = 0;
  for (IntType col = 0; col < n; col += colBlockSize) {
    const IntType currentCols = std::min(n - col, colBlockSize);

    const IntType rowB = opB == SplaOperation::SPLA_OP_NONE ? 0 : col;
    const IntType colB = opB == SplaOperation::SPLA_OP_NONE ? col : 0;
    const IntType numRowsB = opB == SplaOperation::SPLA_OP_NONE ? k : currentCols;
    const IntType numColsB = opB == SplaOperation::SPLA_OP_NONE ? currentCols : k;

    for (IntType row = 0; row < m; row += rowBlockSize, ++counter) {
      const IntType currentRows = std::min(m - row, rowBlockSize);

      const IntType rowA = opA == SplaOperation::SPLA_OP_NONE ? row : 0;
      const IntType colA = opA == SplaOperation::SPLA_OP_NONE ? 0 : row;
      const IntType numRowsA = opA == SplaOperation::SPLA_OP_NONE ? currentRows : k;
      const IntType numColsA = opA == SplaOperation::SPLA_OP_NONE ? k : currentRows;

      const IntType streamIdx = counter % ctx.num_tiles();
      auto viewC = matAccessorsC[streamIdx].get_tile(row, col, currentRows, currentCols,
                                                     blasHandles[streamIdx].stream_handle().get());
      multiply_gpu<T>(blasHandles[streamIdx].get(), opBlasA, opBlasB, alpha,
                      matAccessorsA[streamIdx].sub_accessor(rowA, colA, numRowsA, numColsA),
                      matAccessorsB[streamIdx].sub_accessor(rowB, colB, numRowsB, numColsB), beta,
                      viewC);
      if (hostPtrC) {
        copy_from_gpu_async(
            blasHandles[streamIdx].stream_handle().get(), GPUArrayConstView2D<T>(viewC),
            HostArrayView2D<T>(hostPtrC + col * ldc + row, currentCols, currentRows, ldc));
      }
    }
  }

  for (auto &handle : blasHandles) {
    gpu::check_status(gpu::stream_synchronize(handle.stream_handle().get()));
  }
}

template auto gemm_gpu<float>(SplaOperation opA, SplaOperation opB, IntType m, IntType n, IntType k,
                              float alpha, const float *A, IntType lda, const float *B, IntType ldb,
                              float beta, float *C, IntType ldc, ContextInternal &ctx) -> void;

template auto gemm_gpu<double>(SplaOperation opA, SplaOperation opB, IntType m, IntType n,
                               IntType k, double alpha, const double *A, IntType lda,
                               const double *B, IntType ldb, double beta, double *C, IntType ldc,
                               ContextInternal &ctx) -> void;

template auto gemm_gpu<gpu::blas::ComplexFloatType>(
    SplaOperation opA, SplaOperation opB, IntType m, IntType n, IntType k,
    gpu::blas::ComplexFloatType alpha, const gpu::blas::ComplexFloatType *A, IntType lda,
    const gpu::blas::ComplexFloatType *B, IntType ldb, gpu::blas::ComplexFloatType beta,
    gpu::blas::ComplexFloatType *C, IntType ldc, ContextInternal &ctx) -> void;

template auto gemm_gpu<gpu::blas::ComplexDoubleType>(
    SplaOperation opA, SplaOperation opB, IntType m, IntType n, IntType k,
    gpu::blas::ComplexDoubleType alpha, const gpu::blas::ComplexDoubleType *A, IntType lda,
    const gpu::blas::ComplexDoubleType *B, IntType ldb, gpu::blas::ComplexDoubleType beta,
    gpu::blas::ComplexDoubleType *C, IntType ldc, ContextInternal &ctx) -> void;

}  // namespace spla
