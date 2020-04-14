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
#include <vector>
#include <memory>
#include "gemm_sbs/gemm_sbs_gpu.hpp"
#include "gpu_util/gpu_runtime_api.hpp"
#include "memory/host_array_view.hpp"
#include "spla/context.hpp"
#include "util/common_types.hpp"
#include "util/omp_definitions.hpp"
#include "spla/spla.hpp"
#include "memory/gpu_array_view.hpp"
#include "memory/gpu_array_const_view.hpp"
#include "memory/host_array_const_view.hpp"
#include "gpu_util/gpu_transfer.hpp"
#include "gpu_util/gpu_blas_api.hpp"
#include "spla/context_internal.hpp"
#include "gpu_util/gpu_matrix_accessor.hpp"
#include "gpu_util/gpu_pointer_translation.hpp"
#include "gemm_sbs/stripe_gpu.hpp"
#include "gemm_sbs/multiply_sbs_gpu.hpp"
#include "block_generation/matrix_block_generator.hpp"
#include "block_generation/mirror_generator.hpp"
#include "block_generation/block_cyclic_generator.hpp"

namespace spla {

template <typename T>
static void gemm_sbs_gpu_single_rank(int mLocal, int n, int k, T alpha, const T *A, int lda, const T *B,
                              int ldb, int bRowOffset, int bColOffset,
                              MatrixDistributionInternal &descB, T beta, T *C, int ldc,
                              ContextInternal &ctx) {
  const T* hostPtrA;
  const T* gpuPtrA;
  const T* hostPtrB;
  const T* gpuPtrB;
  T* hostPtrC;
  T* gpuPtrC;

  std::tie(hostPtrA, gpuPtrA) =  translate_gpu_pointer(A);
  std::tie(hostPtrB, gpuPtrB) =  translate_gpu_pointer(B);
  std::tie(hostPtrC, gpuPtrC) =  translate_gpu_pointer(C);

  IntType maxGPUMultiplyBufferSize =
      ctx.gpu_memory_limit() / (sizeof(T) * 3);  // 3 buffers per stripe
  maxGPUMultiplyBufferSize = std::max<IntType>(maxGPUMultiplyBufferSize , 512*512); // miminum of 512^2

  auto& blasHandles = ctx.gpu_blas_handles(1);
  auto& gpuBuffers = ctx.gpu_buffers(3);
  auto matA = gpuPtrA ? GPUMatrixAccessor<GPUArrayConstView2D<T>>(
                            GPUArrayConstView2D<T>(gpuPtrA, k, mLocal, lda))
                      : GPUMatrixAccessor<GPUArrayConstView2D<T>>(
                            HostArrayConstView2D<T>(A, k, mLocal, lda), maxGPUMultiplyBufferSize,
                            gpuBuffers[0]);

  auto matB = gpuPtrB ? GPUMatrixAccessor<GPUArrayConstView2D<T>>(GPUArrayConstView2D<T>(
                            gpuPtrB + bRowOffset + bColOffset * ldb, n, k, ldb))
                      : GPUMatrixAccessor<GPUArrayConstView2D<T>>(
                            HostArrayConstView2D<T>(B + bRowOffset + bColOffset * ldb, n, k, ldb),
                            maxGPUMultiplyBufferSize, gpuBuffers[1]);

  auto matC = gpuPtrC ? GPUMatrixAccessor<GPUArrayView2D<T>>(
                            GPUArrayView2D<T>(gpuPtrC, n, mLocal, ldc))
                      : GPUMatrixAccessor<GPUArrayView2D<T>>(
                            HostArrayConstView2D<T>(C, n, mLocal, ldc), maxGPUMultiplyBufferSize,
                            gpuBuffers[2]);


  IntType rowBlockSize = matC.rows();
  if(matC.max_tile_size() < matC.rows() * matC.cols()) {
    // if not fully on GPU, try square size
    rowBlockSize = std::min<IntType>(std::sqrt(matC.max_tile_size()), rowBlockSize);
  }

  const IntType colBlockSize = std::min(matC.max_tile_size() / rowBlockSize, matC.cols());
  rowBlockSize = std::min(matC.max_tile_size() / colBlockSize, matC.rows());

  for(IntType col =0 ; col < matC.cols(); col += colBlockSize) {
    const IntType currentCols = std::min(matC.cols() - col, colBlockSize);
    for (IntType row = 0; row < matC.rows(); row += rowBlockSize) {
      const IntType currentRows = std::min(matC.rows() - row, rowBlockSize);
      auto viewC = matC.get_tile(row, col, currentRows, currentCols,
                                 blasHandles.front().stream_handle().get());
      multiply_sbs_gpu<T>(blasHandles.front().get(), alpha, matA, matB, beta, viewC);
      if(hostPtrC) {
        copy_from_gpu_async(
            blasHandles.front().stream_handle().get(), GPUArrayConstView2D<T>(viewC),
            HostArrayView2D<T>(hostPtrC + col * ldc + row, currentCols, currentRows, ldc));
      }
    }
  }

  gpu::check_status(gpu::stream_synchronize(blasHandles.front().stream_handle().get()));

}

/*
 *    ------ H     ------
 *    |    |       |    |
 *    |    |       |    |
 *    ------       ------        -------
 *    |    |       |    |        |  |  |
 *    |    |   *   |    |    =   -------
 *    ------       ------        |  |  |
 *    |    |       |    |        -------
 *    |    |       |    |           C
 *    ------       ------
 *    |    |       |    |
 *    |    |       |    |
 *    ------       ------
 *      A            B
 */
template <typename T>
void gemm_sbs_gpu(int mLocal, int n, int k, T alpha, const T *A, int lda, const T *B,
                  int ldb, int bRowOffset, int bColOffset, MatrixDistributionInternal &descB,
                  T beta, T *C, int ldc, ContextInternal &ctx) {
  if(n == 0 || k == 0) {
    return;
  }

  if(descB.comm().size() == 1 || descB.type() == SplaDistributionType::SPLA_DIST_MIRROR) {
    return gemm_sbs_gpu_single_rank<T>(mLocal, n, k, alpha, A, lda, B, ldb, bRowOffset, bColOffset,
                                       descB, beta, C, ldc, ctx);
  }

  std::shared_ptr<MatrixBlockGenerator> matrixDist;
  if (descB.type() == SplaDistributionType::SPLA_DIST_BLACS_BLOCK_CYCLIC) {
    matrixDist.reset(new BlockCyclicGenerator(descB.row_block_size(), descB.col_block_size(),
                                                 descB.proc_grid_rows(), descB.proc_grid_cols(),
                                                 k, n, bRowOffset, bColOffset));
  } else {
    matrixDist.reset(new MirrorGenerator(ctx.tile_length_target(), ctx.tile_length_target(), k,
                                              n, bRowOffset, bColOffset));
  }
  const IntType numBlockRows = matrixDist->num_block_rows();
  const IntType numBlockCols = matrixDist->num_block_cols();

  const IntType numBlockColsInTile =
      std::max<IntType>((ctx.tile_length_target() + descB.col_block_size() - 1) / descB.col_block_size(), 1);

  // calcualte memory target
  IntType maxGPUMultiplyBufferSize =
      ctx.gpu_memory_limit() / (ctx.num_gpu_streams() * sizeof(T) * 3);  // 3 buffers per stripe
  maxGPUMultiplyBufferSize = std::max<IntType>(maxGPUMultiplyBufferSize , 512*512); // miminum of 512^2

  std::vector<StripeGPU<T>> stripes;
  stripes.reserve(ctx.num_gpu_streams());

  auto& gpuBuffers = ctx.gpu_buffers(ctx.num_gpu_streams() * 3);
  auto &pinnedBuffers = ctx.pinned_buffers(2 * ctx.num_gpu_streams());
  auto& blasHandles = ctx.gpu_blas_handles(ctx.num_gpu_streams());

  const T* hostPtrA;
  const T* gpuPtrA;
  const T* hostPtrB;
  const T* gpuPtrB;
  T* hostPtrC;
  T* gpuPtrC;

  std::tie(hostPtrA, gpuPtrA) =  translate_gpu_pointer(A);
  std::tie(hostPtrB, gpuPtrB) =  translate_gpu_pointer(B);
  std::tie(hostPtrC, gpuPtrC) =  translate_gpu_pointer(C);

  for (IntType i = 0; i < ctx.num_gpu_streams(); ++i) {
    auto matA = gpuPtrA ? GPUMatrixAccessor<GPUArrayConstView2D<T>>(
                              GPUArrayConstView2D<T>(gpuPtrA, k, mLocal, lda))
                        : GPUMatrixAccessor<GPUArrayConstView2D<T>>(
                              HostArrayConstView2D<T>(A, k, mLocal, lda), maxGPUMultiplyBufferSize,
                              gpuBuffers[i * 3]);

    auto matC = gpuPtrC ? GPUMatrixAccessor<GPUArrayView2D<T>>(
                              GPUArrayView2D<T>(gpuPtrC, n, mLocal, ldc))
                        : GPUMatrixAccessor<GPUArrayView2D<T>>(
                              HostArrayView2D<T>(C, n, mLocal, ldc), maxGPUMultiplyBufferSize,
                              gpuBuffers[i * 3 + 1]);

    auto hostMatC =
        gpuPtrC ? HostArrayView2D<T>() : HostArrayView2D<T>(C, n, mLocal, ldc);

    auto hostMatB =
        gpuPtrB ? HostArrayConstView2D<T>() : HostArrayConstView2D<T>(B, n + bColOffset, ldb);
    auto gpuMatB =
        gpuPtrB ? GPUArrayConstView2D<T>() : GPUArrayConstView2D<T>(B, n + bColOffset, ldb);

    stripes.emplace_back(descB.comm(), blasHandles[i], pinnedBuffers[2 * i],
                         pinnedBuffers[2 * i + 1], gpuBuffers[i * 3 + 2], maxGPUMultiplyBufferSize,
                         matrixDist, alpha, matA, hostMatB, gpuMatB, beta, matC, hostMatC,
                         numBlockColsInTile);
  }

  if (ctx.num_threads() > 1) {
    // comm + worker thread
    SPLA_OMP_PRAGMA("omp parallel num_threads(2)") {
      // TODO set device per thread
      IntType counter = 0;
      for (IntType blockColIdx = 0; blockColIdx < numBlockCols;
           blockColIdx += numBlockColsInTile, ++counter) {
        auto &t = stripes[counter % ctx.num_gpu_streams()];
        auto &tNext = stripes[(counter + 1) % ctx.num_gpu_streams()];
        if (omp_get_thread_num() == 0) {
          // wait for tile to be multiplied
          while (t.state() != StripeState::Collected) {
          }
          t.start_exchange();
          t.finalize_exchange();
        } else {
          // wait for tile once encountering the same tile more than once
          if (counter >= ctx.num_gpu_streams() - 1) {
            while (tNext.state() != StripeState::Exchanged) {
            }
            tNext.multiply();
          }
          t.collect(blockColIdx);
        }
      }
    }
  } else {
    // single thread
    IntType counter = 0;
    for (IntType blockColIdx = 0; blockColIdx < numBlockCols;
         blockColIdx += numBlockColsInTile, ++counter) {
      auto &t = stripes[counter % ctx.num_gpu_streams()];
      auto &tNext = stripes[(counter + 1) % ctx.num_gpu_streams()];

      if (tNext.state() == StripeState::InExchange) {
        tNext.finalize_exchange();
        tNext.multiply();
      }

      t.collect(blockColIdx);
      t.start_exchange();
    }
  }

  // finalize remaining stripes
  for(auto& t: stripes) {
    if (t.state() == StripeState::InExchange) {
      t.finalize_exchange();
    }
    if (t.state() == StripeState::Exchanged) {
      t.multiply();
    }
  }
  for(auto& t: stripes) {
    t.synchronize();
  }
}

template void gemm_sbs_gpu<float>(int mLocal, int n, int k, float alpha, const float *A, int lda,
                                  const float *B, int ldb, int bRowOffset, int bColOffset,
                                  MatrixDistributionInternal &descB, float beta, float *C, int ldc,
                                  ContextInternal &ctx);

template void gemm_sbs_gpu<double>(int mLocal, int n, int k, double alpha, const double *A, int lda,
                                  const double *B, int ldb, int bRowOffset, int bColOffset,
                                  MatrixDistributionInternal &descB, double beta, double *C, int ldc,
                                  ContextInternal &ctx);

template void gemm_sbs_gpu<gpu::blas::ComplexFloatType>(
    int mLocal, int n, int k, gpu::blas::ComplexFloatType alpha,
    const gpu::blas::ComplexFloatType *A, int lda, const gpu::blas::ComplexFloatType *B, int ldb,
    int bRowOffset, int bColOffset, MatrixDistributionInternal &descB,
    gpu::blas::ComplexFloatType beta, gpu::blas::ComplexFloatType *C, int ldc,
    ContextInternal &ctx);

template void gemm_sbs_gpu<gpu::blas::ComplexDoubleType>(
    int mLocal, int n, int k, gpu::blas::ComplexDoubleType alpha,
    const gpu::blas::ComplexDoubleType *A, int lda, const gpu::blas::ComplexDoubleType *B, int ldb,
    int bRowOffset, int bColOffset, MatrixDistributionInternal &descB,
    gpu::blas::ComplexDoubleType beta, gpu::blas::ComplexDoubleType *C, int ldc,
    ContextInternal &ctx);

}  // namespace spla
