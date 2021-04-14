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
#include "pgemm_sbs/ring_sbs_gpu.hpp"
#include "pgemm_ssb/block_size_selection_ssb.hpp"
#include "spla/context.hpp"
#include "spla/context_internal.hpp"
#include "spla/spla.hpp"
#include "timing/timing.hpp"
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


  const T *hostPtrA;
  const T *gpuPtrA;
  const T *hostPtrB;
  const T *gpuPtrB;
  T *hostPtrC;
  T *gpuPtrC;

  std::tie(hostPtrA, gpuPtrA) = translate_gpu_pointer(A);
  std::tie(hostPtrB, gpuPtrB) = translate_gpu_pointer(B);
  std::tie(hostPtrC, gpuPtrC) = translate_gpu_pointer(C);

  const IntType tileSizeGEMM = ctx.tile_size_gpu() * ctx.tile_size_gpu();

  /*************************************
   * Try to determine optimal block size
   *************************************/
  IntType rowsInBlock = 1;
  IntType colsInBlock = 1;

  const double ringThreshold = 0.65;
  const IntType minBlockSize =
      gpuPtrA && gpuPtrB
          ? 250
          : 500;  // If input is on host, smal block sizes lead to much more memory transfers
                  // required. Therefore use larger block sizes in that case.

  std::tie(rowsInBlock, colsInBlock) = block_size_selection_ssb(
      SPLA_FILL_MODE_FULL, IsDisjointGenerator<BLOCK_GEN>::value, 1.0 - ringThreshold,
      descB.comm().size(), k, n, bRowOffset, bColOffset, ctx.tile_size_host(), minBlockSize);

  // Compute maximum block sizes such that memory allocations for increasing m / n can be avoided
  const IntType maxBlockSize =
      std::max<IntType>(rowsInBlock * colsInBlock, ctx.tile_size_host() * ctx.tile_size_host());

  const IntType numRingProcs = 2;
  const IntType numTiles =
      std::max<IntType>(1, (ctx.num_tiles() + numRingProcs - 1) / numRingProcs);

  /*************************************
   * Create tiles
   *************************************/

  auto pinnedBuffersIt = ctx.pinned_buffers(numTiles * (numRingProcs + 1)).begin();
  auto gpuBuffersIt = ctx.gpu_buffers(numTiles * numRingProcs * 3).begin();
  auto blasHandlesIt = ctx.gpu_blas_handles(numTiles * numRingProcs).begin();
  auto commsIt = descB.get_comms(numTiles).begin();

  std::vector<RingSBSGPU<T, BLOCK_GEN>> tiles;
  tiles.reserve(numTiles);

  auto hostMatB = gpuPtrB ? HostArrayConstView2D<T>() : HostArrayConstView2D<T>(B, n + bColOffset, ldb, ldb);

  auto gpuMatB =
      gpuPtrB ? GPUArrayConstView2D<T>(gpuPtrB, n + bColOffset, ldb, ldb) : GPUArrayConstView2D<T>();

  for (IntType i = 0; i < numTiles; ++i) {
    std::vector<RingProcessorSBS<T>> ringBlocks;
    ringBlocks.reserve(numTiles * numRingProcs);
    for (IntType j = 0; j < numRingProcs; ++j) {
      auto matA = gpuPtrA
                      ? GPUConstMatrixAccessor<T>(GPUArrayConstView2D<T>(gpuPtrA, k, mLocal, lda))
                      : GPUConstMatrixAccessor<T>(HostArrayConstView2D<T>(A, k, mLocal, lda),
                                                  tileSizeGEMM, *(gpuBuffersIt++));

      auto matC = gpuPtrC ? GPUMatrixAccessor<T>(GPUArrayView2D<T>(gpuPtrC, n, mLocal, ldc))
                          : GPUMatrixAccessor<T>(HostArrayView2D<T>(C, n, mLocal, ldc),
                                                 tileSizeGEMM, *(gpuBuffersIt++));

      ringBlocks.emplace_back(maxBlockSize, *(blasHandlesIt++), *(pinnedBuffersIt++),
                              *(gpuBuffersIt++), std::move(matA), std::move(matC));
    }

    tiles.emplace_back(ringThreshold, maxBlockSize, *(commsIt++), std::move(ringBlocks), gen, alpha,
                       beta, hostMatB, gpuMatB, bRowOffset, bColOffset);
  }

  /*************************************
   * Start processing
   *************************************/

  std::vector<Block> blocks;
  blocks.reserve(descB.comm().size());
  std::unordered_set<IntType> betaColIndeces;
  auto& colEvents = ctx.gpu_event_handles(20);

  IntType tileIdx = 0;

  // iterate grid wise
  for (IntType colStartIdx = 0; colStartIdx < n;
       colStartIdx += descB.proc_grid_cols() * colsInBlock) {
    for (IntType rowStartIdx = 0; rowStartIdx < k;
         rowStartIdx += descB.proc_grid_rows() * rowsInBlock) {
      // iterate through blocks within grid
      for (IntType colIdx = colStartIdx;
           colIdx < std::min<IntType>(n, colStartIdx + descB.proc_grid_cols() * colsInBlock);
           colIdx += colsInBlock) {
        for (IntType rowIdx = rowStartIdx;
             rowIdx < std::min<IntType>(k, rowStartIdx + descB.proc_grid_rows() * rowsInBlock);
             rowIdx += rowsInBlock) {
          const auto block = Block{rowIdx, colIdx, std::min<IntType>(rowsInBlock, k - rowIdx),
                                   std::min<IntType>(colsInBlock, n - colIdx)};
          blocks.emplace_back(block);
          // Prepare processing when there are enough blocks to form ring
          if (blocks.size() == descB.comm().size()) {
            auto &t = tiles[tileIdx % numTiles];
            t.prepare(blocks.begin(), blocks.end());
            blocks.resize(0);
            ++tileIdx;
            t.process_step(betaColIndeces,
                           colEvents);  // do one step for better comm / compute overlap
          }

          if (tileIdx == numTiles) {
            // All tiles are prepared -> start processing
            bool tileToProcess = true;
            while (tileToProcess) {
              tileToProcess = false;
              // Interleave processing to hide communication cost
              for (auto &t : tiles) {
                tileToProcess |= t.process_step(betaColIndeces, colEvents);
              }
            }
            tileIdx = 0;
          }
        }
      }
    }
  }

  if (blocks.size()) {
    // Prepare with remaining blocks
    auto &t = tiles[tileIdx];
    t.prepare(blocks.begin(), blocks.end());
    t.process_step(betaColIndeces, colEvents);  // do one step for better comm / compute overlap
    blocks.resize(0);
  }
  // Process remaining blocks
  bool tileToProcess = true;
  while (tileToProcess) {
    tileToProcess = false;
    for (auto &t : tiles) {
      tileToProcess |= t.process_step(betaColIndeces, colEvents);
    }
  }

  // synchronize all streams
  for (auto &t : tiles) {
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
