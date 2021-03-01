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
#include "pgemm_sbs/pgemm_sbs_gpu.hpp"

#include <algorithm>
#include <memory>
#include <vector>

#include "block_generation/block_cyclic_generator.hpp"
#include "block_generation/mirror_generator.hpp"
#include "gemm/gemm_gpu.hpp"
#include "gpu_util/gpu_blas_api.hpp"
#include "gpu_util/gpu_device_guard.hpp"
#include "gpu_util/gpu_matrix_accessor.hpp"
#include "gpu_util/gpu_pointer_translation.hpp"
#include "gpu_util/gpu_runtime_api.hpp"
#include "gpu_util/gpu_transfer.hpp"
#include "memory/gpu_array_const_view.hpp"
#include "memory/gpu_array_view.hpp"
#include "memory/host_array_const_view.hpp"
#include "memory/host_array_view.hpp"
#include "pgemm_sbs/stripe_gpu.hpp"
#include "spla/context.hpp"
#include "spla/context_internal.hpp"
#include "spla/spla.hpp"
#include "util/check_gemm_param.hpp"
#include "util/common_types.hpp"
#include "util/omp_definitions.hpp"

namespace spla {
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
template <typename T, typename BLOCK_GEN>
void pgemm_sbs_gpu_internal(int mLocal, int n, int k, T alpha, const T *A, int lda, const T *B,
                            int ldb, int bRowOffset, int bColOffset,
                            MatrixDistributionInternal &descB, T beta, T *C, int ldc,
                            ContextInternal &ctx, BLOCK_GEN gen) {
  check_gemm_param(SplaOperation::SPLA_OP_NONE, SplaOperation::SPLA_OP_NONE, mLocal,
                   gen.local_cols(descB.comm().rank()), gen.local_rows(descB.comm().rank()), A, lda,
                   B, ldb, C, ldc);

  GPUDeviceGuard deviceGuard(ctx.gpu_device_id());

  // always synchronize with stream 0 as part of API requirement
  gpu::check_status(gpu::stream_synchronize(nullptr));

  const IntType numBlockRows = gen.num_block_rows();
  const IntType numBlockCols = gen.num_block_cols();

  const IntType numBlockColsInTile = std::max<IntType>(
      (ctx.tile_size_host() + descB.col_block_size() - 1) / descB.col_block_size(), 1);

  const IntType tileSizeGEMM = ctx.tile_size_gpu() * ctx.tile_size_gpu();

  std::vector<StripeGPU<T, BLOCK_GEN>> stripes;
  stripes.reserve(ctx.num_tiles());

  auto &gpuBuffers = ctx.gpu_buffers(ctx.num_tiles() * 3);
  auto &pinnedBuffers = ctx.pinned_buffers(2 * ctx.num_tiles());
  auto &blasHandles = ctx.gpu_blas_handles(ctx.num_tiles());

  const T *hostPtrA;
  const T *gpuPtrA;
  const T *hostPtrB;
  const T *gpuPtrB;
  T *hostPtrC;
  T *gpuPtrC;

  std::tie(hostPtrA, gpuPtrA) = translate_gpu_pointer(A);
  std::tie(hostPtrB, gpuPtrB) = translate_gpu_pointer(B);
  std::tie(hostPtrC, gpuPtrC) = translate_gpu_pointer(C);

  for (IntType i = 0; i < ctx.num_tiles(); ++i) {
    auto matA =
        gpuPtrA ? GPUMatrixAccessor<GPUArrayConstView2D<T>>(
                      GPUArrayConstView2D<T>(gpuPtrA, k, mLocal, lda))
                : GPUMatrixAccessor<GPUArrayConstView2D<T>>(
                      HostArrayConstView2D<T>(A, k, mLocal, lda), tileSizeGEMM, gpuBuffers[i * 3]);

    auto matC =
        gpuPtrC ? GPUMatrixAccessor<GPUArrayView2D<T>>(GPUArrayView2D<T>(gpuPtrC, n, mLocal, ldc))
                : GPUMatrixAccessor<GPUArrayView2D<T>>(HostArrayView2D<T>(C, n, mLocal, ldc),
                                                       tileSizeGEMM, gpuBuffers[i * 3 + 1]);

    auto hostMatC = gpuPtrC ? HostArrayView2D<T>() : HostArrayView2D<T>(C, n, mLocal, ldc);

    auto hostMatB =
        gpuPtrB ? HostArrayConstView2D<T>() : HostArrayConstView2D<T>(B, n + bColOffset, ldb);
    auto gpuMatB =
        gpuPtrB ? GPUArrayConstView2D<T>(B, n + bColOffset, ldb) : GPUArrayConstView2D<T>();

    stripes.emplace_back(descB.comm(), blasHandles[i], pinnedBuffers[2 * i],
                         pinnedBuffers[2 * i + 1], gpuBuffers[i * 3 + 2], ctx.tile_size_gpu(), gen,
                         alpha, matA, hostMatB, gpuMatB, beta, matC, hostMatC, numBlockColsInTile);
  }

  if (ctx.num_threads() > 1) {
    // comm + worker thread
    SPLA_OMP_PRAGMA("omp parallel num_threads(2)") {
      GPUDeviceGuard deviceGuard(ctx.gpu_device_id());
      IntType counter = 0;
      for (IntType blockColIdx = 0; blockColIdx < numBlockCols;
           blockColIdx += numBlockColsInTile, ++counter) {
        auto &t = stripes[counter % ctx.num_tiles()];
        auto &tNext = stripes[(counter + 1) % ctx.num_tiles()];
        if (omp_get_thread_num() == 0) {
          // wait for tile to be multiplied
          while (t.state() != StripeState::Collected) {
          }
          t.start_exchange();
          t.finalize_exchange();
        } else {
          // wait for tile once encountering the same tile more than once
          if (counter >= ctx.num_tiles() - 1) {
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
      auto &t = stripes[counter % ctx.num_tiles()];
      auto &tNext = stripes[(counter + 1) % ctx.num_tiles()];

      if (tNext.state() == StripeState::InExchange) {
        tNext.finalize_exchange();
        tNext.multiply();
      }

      t.collect(blockColIdx);
      t.start_exchange();
    }
  }

  // finalize remaining stripes
  for (auto &t : stripes) {
    if (t.state() == StripeState::InExchange) {
      t.finalize_exchange();
    }
    if (t.state() == StripeState::Exchanged) {
      t.multiply();
    }
  }
  for (auto &t : stripes) {
    t.synchronize();
  }
}

template <typename T>
void pgemm_sbs_gpu(int mLocal, int n, int k, T alpha, const T *A, int lda, const T *B, int ldb,
                   int bRowOffset, int bColOffset, MatrixDistributionInternal &descB, T beta, T *C,
                   int ldc, ContextInternal &ctx) {
  if (n == 0 || k == 0) {
    return;
  }

  if (n < 0 || k < 0 || bRowOffset < 0 || bColOffset < 0) {
    throw InvalidParameterError();
  }

  if (descB.comm().size() == 1 || descB.type() == SplaDistributionType::SPLA_DIST_MIRROR) {
    return gemm_gpu<T>(SplaOperation::SPLA_OP_NONE, SplaOperation::SPLA_OP_NONE, mLocal, n, k,
                       alpha, A, lda, B + bRowOffset + bColOffset * ldb, ldb, beta, C, ldc, ctx);
  }

  BlockCyclicGenerator gen(descB.row_block_size(), descB.col_block_size(), descB.proc_grid_rows(),
                           descB.proc_grid_cols(), k, n, bRowOffset, bColOffset);

  pgemm_sbs_gpu_internal<T, BlockCyclicGenerator>(mLocal, n, k, alpha, A, lda, B, ldb, bRowOffset,
                                                  bColOffset, descB, beta, C, ldc, ctx,
                                                  std::move(gen));
}

template void pgemm_sbs_gpu<float>(int mLocal, int n, int k, float alpha, const float *A, int lda,
                                   const float *B, int ldb, int bRowOffset, int bColOffset,
                                   MatrixDistributionInternal &descB, float beta, float *C, int ldc,
                                   ContextInternal &ctx);

template void pgemm_sbs_gpu<double>(int mLocal, int n, int k, double alpha, const double *A,
                                    int lda, const double *B, int ldb, int bRowOffset,
                                    int bColOffset, MatrixDistributionInternal &descB, double beta,
                                    double *C, int ldc, ContextInternal &ctx);

template void pgemm_sbs_gpu<gpu::blas::ComplexFloatType>(
    int mLocal, int n, int k, gpu::blas::ComplexFloatType alpha,
    const gpu::blas::ComplexFloatType *A, int lda, const gpu::blas::ComplexFloatType *B, int ldb,
    int bRowOffset, int bColOffset, MatrixDistributionInternal &descB,
    gpu::blas::ComplexFloatType beta, gpu::blas::ComplexFloatType *C, int ldc,
    ContextInternal &ctx);

template void pgemm_sbs_gpu<gpu::blas::ComplexDoubleType>(
    int mLocal, int n, int k, gpu::blas::ComplexDoubleType alpha,
    const gpu::blas::ComplexDoubleType *A, int lda, const gpu::blas::ComplexDoubleType *B, int ldb,
    int bRowOffset, int bColOffset, MatrixDistributionInternal &descB,
    gpu::blas::ComplexDoubleType beta, gpu::blas::ComplexDoubleType *C, int ldc,
    ContextInternal &ctx);

}  // namespace spla
