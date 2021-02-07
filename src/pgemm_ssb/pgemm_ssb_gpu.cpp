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
#include <memory>
#include <vector>
#include "block_generation/block_cyclic_generator.hpp"
#include "block_generation/matrix_block_generator.hpp"
#include "block_generation/mirror_generator.hpp"
#include "gemm/gemm_gpu.hpp"
#include "gpu_util/gpu_blas_api.hpp"
#include "gpu_util/gpu_matrix_accessor.hpp"
#include "gpu_util/gpu_pointer_translation.hpp"
#include "gpu_util/gpu_runtime_api.hpp"
#include "gpu_util/gpu_transfer.hpp"
#include "gpu_util/gpu_device_guard.hpp"
#include "memory/gpu_array_const_view.hpp"
#include "memory/gpu_array_view.hpp"
#include "memory/host_array_const_view.hpp"
#include "memory/host_array_view.hpp"
#include "pgemm_ssb/ring_reduce_tile_gpu.hpp"
#include "spla/context.hpp"
#include "spla/context_internal.hpp"
#include "spla/matrix_distribution_internal.hpp"
#include "spla/spla.hpp"
#include "util/common_types.hpp"
#include "util/omp_definitions.hpp"
#include "util/check_gemm_param.hpp"

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
void pgemm_ssb_gpu_internal(int m, int n, int kLocal, SplaOperation opA,
                            T alpha, const T *A, int lda, const T *B, int ldb,
                            T beta, T *C, int ldc, int cRowStart, int cColStart,
                            MatrixDistributionInternal &descC,
                            ContextInternal &ctx, BLOCK_GEN gen) {

  check_gemm_param(
      opA, SplaOperation::SPLA_OP_NONE, gen.local_rows(descC.comm().rank()),
      gen.local_cols(descC.comm().rank()), kLocal, A, lda, B, ldb, C, ldc);

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

  const IntType numTiles = 2;
  const IntType numRingProcs = 2;

  auto pinnedBuffersIt =
      ctx.pinned_buffers(numTiles * (numRingProcs + 1)).begin();
  auto gpuBuffersIt = ctx.gpu_buffers(numTiles * numRingProcs * 3).begin();
  auto blasHandlesIt = ctx.gpu_blas_handles(numTiles * numRingProcs).begin();
  auto eventHandlesIt = ctx.gpu_event_handles(numTiles * numRingProcs).begin();
  auto streamHandlesIt =
      ctx.gpu_stream_handles(numTiles * numRingProcs).begin();
  auto commsIt = descC.get_comms(numTiles).begin();

  std::vector<RingReduceTileGPU<T, BLOCK_GEN>> tiles;
  tiles.reserve(numTiles);

  auto hostMatC = gpuPtrC ? HostArrayView2D<T>()
                          : HostArrayView2D<T>(C, n + cColStart, ldc, ldc);

  auto gpuMatC = gpuPtrC ? GPUArrayView2D<T>(gpuPtrC, n + cColStart, ldc, ldc)
                         : GPUArrayView2D<T>();

  const IntType rowsInBlock = gen.max_rows_in_block();
  const IntType colsInBlock = gen.max_cols_in_block();

  for (IntType i = 0; i < numTiles; ++i) {
    std::vector<RingProcessor<T>> ringBlocks;
    ringBlocks.reserve(numTiles * numRingProcs);
    for (IntType j = 0; j < numRingProcs; ++j) {
      auto matA = gpuPtrA ? GPUMatrixAccessor<GPUArrayConstView2D<T>>(
                                GPUArrayConstView2D<T>(gpuPtrA, m, kLocal, lda))
                          : GPUMatrixAccessor<GPUArrayConstView2D<T>>(
                                HostArrayConstView2D<T>(A, m, kLocal, lda),
                                tileSizeGEMM, *(gpuBuffersIt++));

      auto matB = gpuPtrB ? GPUMatrixAccessor<GPUArrayConstView2D<T>>(
                                GPUArrayConstView2D<T>(gpuPtrB, n, kLocal, ldb))
                          : GPUMatrixAccessor<GPUArrayConstView2D<T>>(
                                HostArrayConstView2D<T>(B, n, kLocal, ldb),
                                tileSizeGEMM, *(gpuBuffersIt++));
      ringBlocks.emplace_back(
          rowsInBlock * colsInBlock, *(blasHandlesIt++),
          *(eventHandlesIt++), *(streamHandlesIt++), *(pinnedBuffersIt++),
          *(gpuBuffersIt++), std::move(matA), std::move(matB));
    }

    tiles.emplace_back(rowsInBlock * colsInBlock, *(commsIt++),
                       std::move(ringBlocks), *(pinnedBuffersIt++), gen, opA,
                       alpha, beta, hostMatC, gpuMatC);
  }

  std::vector<BlockCoord> blocks;
  blocks.reserve(descC.comm().size());


  IntType tileIdx = 0;

  // iterate grid wise
  for (IntType colStartIdx = 0; colStartIdx < n;
       colStartIdx += descC.proc_grid_cols() * colsInBlock) {
    for (IntType rowStartIdx = 0; rowStartIdx < m;
         rowStartIdx += descC.proc_grid_rows() * rowsInBlock) {

      // iterate through blocks within grid
      for (IntType colIdx = colStartIdx;
           colIdx < std::min<IntType>(n, colStartIdx + descC.proc_grid_cols() *
                                                           colsInBlock);
           colIdx += colsInBlock) {
        for (IntType rowIdx = rowStartIdx;
             rowIdx <
             std::min<IntType>(m, rowStartIdx +
                                      descC.proc_grid_rows() * rowsInBlock);
             rowIdx += rowsInBlock) {

          blocks.emplace_back(BlockCoord{
              rowIdx, colIdx, std::min<IntType>(rowsInBlock, m - rowIdx),
              std::min<IntType>(colsInBlock, n - colIdx)});

          // Prepare processing when there are enough blocks to form ring
          if (blocks.size() == descC.comm().size()) {
            auto &t = tiles[tileIdx % numTiles];
            assert(t.state() == TileState::Processed ||
                   t.state() == TileState::Empty);
            if (t.state() == TileState::Processed)
              t.finalize();
            t.prepare(blocks.begin(), blocks.end());
            blocks.resize(0);
            ++tileIdx;
          }

          if (tileIdx == numTiles) {
            // All tiles are prepared -> start processing
            bool tileToProcess = true;
            while (tileToProcess) {
              tileToProcess = false;
              // Interleave processing to hide communication cost
              for (auto &t : tiles) {
                tileToProcess |= t.process_step();
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
    if (tiles[tileIdx].state() == TileState::Processed)
      tiles[tileIdx].finalize();
    tiles[tileIdx].prepare(blocks.begin(), blocks.end());
    blocks.resize(0);
  }

  // Process remaining blocks
  bool tileToProcess = true;
  while (tileToProcess) {
    tileToProcess = false;
    for (auto &t : tiles) {
      tileToProcess |= t.process_step();
    }
  }

  for (auto &t : tiles) {
    if (t.state() == TileState::Processed)
      t.finalize();
  }

  // synchronize all streams
  for (auto &t : tiles) {
    t.synchronize();
  }
}

template <typename T>
void pgemm_ssb_gpu(int m, int n, int kLocal, SplaOperation opA, T alpha,
                   const T *A, int lda, const T *B, int ldb, T beta, T *C,
                   int ldc, int cRowStart, int cColStart,
                   MatrixDistributionInternal &descC, ContextInternal &ctx) {
  if (m == 0 || n == 0) {
    return;
  }

  if (opA != SplaOperation::SPLA_OP_TRANSPOSE &&
      opA != SplaOperation::SPLA_OP_CONJ_TRANSPOSE) {
    throw InvalidParameterError();
  }

  if (m < 0 || n < 0 || cRowStart < 0 || cColStart < 0) {
    throw InvalidParameterError();
  }

  if (descC.comm().size() == 1) {
    return gemm_gpu<T>(opA, SplaOperation::SPLA_OP_NONE, m, n, kLocal, alpha, A,
                       lda, B, ldb, beta, C + cRowStart + cColStart * ldc, ldc,
                       ctx);
  }

  if (descC.type() == SplaDistributionType::SPLA_DIST_BLACS_BLOCK_CYCLIC) {
    BlockCyclicGenerator gen(descC.row_block_size(), descC.col_block_size(),
                             descC.proc_grid_rows(), descC.proc_grid_cols(), m,
                             n, cRowStart, cColStart);

    pgemm_ssb_gpu_internal<T, BlockCyclicGenerator>(
        m, n, kLocal, opA, alpha, A, lda, B, ldb, beta, C, ldc, cRowStart,
        cColStart, descC, ctx, std::move(gen));
  } else {
    MirrorGenerator gen(ctx.tile_size_host(), ctx.tile_size_host(), m, n,
                        cRowStart, cColStart);
    pgemm_ssb_gpu_internal<T, MirrorGenerator>(
        m, n, kLocal, opA, alpha, A, lda, B, ldb, beta, C, ldc, cRowStart,
        cColStart, descC, ctx, std::move(gen));
  }
}

template void pgemm_ssb_gpu<float>(int m, int n, int kLocal, SplaOperation opA,
                                   float alpha, const float *A, int lda,
                                   const float *B, int ldb, float beta,
                                   float *C, int ldc, int cRowStart,
                                   int cColStart,
                                   MatrixDistributionInternal &descC,
                                   ContextInternal &ctx);

template void pgemm_ssb_gpu<double>(int m, int n, int kLocal, SplaOperation opA,
                                    double alpha, const double *A, int lda,
                                    const double *B, int ldb, double beta,
                                    double *C, int ldc, int cRowStart,
                                    int cColStart,
                                    MatrixDistributionInternal &descC,
                                    ContextInternal &ctx);

template void pgemm_ssb_gpu<gpu::blas::ComplexFloatType>(
    int m, int n, int kLocal, SplaOperation opA,
    gpu::blas::ComplexFloatType alpha, const gpu::blas::ComplexFloatType *A,
    int lda, const gpu::blas::ComplexFloatType *B, int ldb,
    gpu::blas::ComplexFloatType beta, gpu::blas::ComplexFloatType *C, int ldc,
    int cRowStart, int cColStart, MatrixDistributionInternal &descC,
    ContextInternal &ctx);

template void pgemm_ssb_gpu<gpu::blas::ComplexDoubleType>(
    int m, int n, int kLocal, SplaOperation opA,
    gpu::blas::ComplexDoubleType alpha, const gpu::blas::ComplexDoubleType *A,
    int lda, const gpu::blas::ComplexDoubleType *B, int ldb,
    gpu::blas::ComplexDoubleType beta, gpu::blas::ComplexDoubleType *C, int ldc,
    int cRowStart, int cColStart, MatrixDistributionInternal &descC,
    ContextInternal &ctx);

} // namespace spla
