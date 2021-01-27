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
#include "pgemm_ssb/all_reduce_tile_gpu.hpp"
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
template <typename T>
void pgemm_ssb_gpu(int m, int n, int kLocal, SplaOperation opA, T alpha, const T *A, int lda,
                   const T *B, int ldb, T beta, T *C, int ldc, int cRowStart, int cColStart,
                   MatrixDistributionInternal &descC, ContextInternal &ctx) {
  if (m == 0 || n == 0) {
    return;
  }

  if (opA != SplaOperation::SPLA_OP_TRANSPOSE && opA != SplaOperation::SPLA_OP_CONJ_TRANSPOSE) {
    throw InvalidParameterError();
  }

  if (m < 0 || n < 0 || cRowStart < 0 || cColStart < 0) {
    throw InvalidParameterError();
  }

  // if (descC.comm().size() == 1) {
  //   return gemm_gpu<T>(opA, SplaOperation::SPLA_OP_NONE, m, n, kLocal, alpha, A, lda, B, ldb, beta,
  //                      C + cRowStart + cColStart * ldc, ldc, ctx);
  // }

  bool useRingReduce = false;
  std::shared_ptr<MatrixBlockGenerator> matrixDist;
  if (descC.type() == SplaDistributionType::SPLA_DIST_BLACS_BLOCK_CYCLIC) {
    matrixDist.reset(new BlockCyclicGenerator(descC.row_block_size(), descC.col_block_size(),
                                              descC.proc_grid_rows(), descC.proc_grid_cols(), m, n,
                                              cRowStart, cColStart));
    // Use ring reduce if block size is at least a given size and most ranks
    // hold result blocks
    if (descC.row_block_size() * descC.col_block_size() >= 256 * 256 &&
        descC.comm().size() >
            (descC.proc_grid_rows() * descC.proc_grid_cols()) / 2) {
      useRingReduce = true;
    }
  } else {
    matrixDist.reset(new MirrorGenerator(ctx.tile_size_host(), ctx.tile_size_host(), m, n,
                                         cRowStart, cColStart));
  }

  check_gemm_param(opA, SplaOperation::SPLA_OP_NONE, matrixDist->local_rows(descC.comm().rank()),
                   matrixDist->local_cols(descC.comm().rank()), kLocal, A, lda, B, ldb, C, ldc);

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
  const IntType numTiles = ctx.num_tiles();

  auto &gpuBuffers = ctx.gpu_buffers(numTiles * 3);
  auto &blasHandles = ctx.gpu_blas_handles(numTiles);

  if (useRingReduce) {
    auto &pinnedBuffers = ctx.pinned_buffers(2 * numTiles);
    std::vector<RingReduceTileGPU<T>> tiles;
    tiles.reserve(numTiles);
    auto &comms = descC.get_comms(numTiles);

    for (IntType i = 0; i < numTiles; ++i) {
      auto matA = gpuPtrA ? GPUMatrixAccessor<GPUArrayConstView2D<T>>(
                                GPUArrayConstView2D<T>(gpuPtrA, m, kLocal, lda))
                          : GPUMatrixAccessor<GPUArrayConstView2D<T>>(
                                HostArrayConstView2D<T>(A, m, kLocal, lda),
                                tileSizeGEMM, gpuBuffers[i * 3]);

      auto matB = gpuPtrB ? GPUMatrixAccessor<GPUArrayConstView2D<T>>(
                                GPUArrayConstView2D<T>(gpuPtrB, n, kLocal, ldb))
                          : GPUMatrixAccessor<GPUArrayConstView2D<T>>(
                                HostArrayConstView2D<T>(B, n, kLocal, ldb),
                                tileSizeGEMM, gpuBuffers[i * 3 + 1]);

      auto hostMatC = gpuPtrC ? HostArrayView2D<T>()
                              : HostArrayView2D<T>(C, n + cColStart, ldc, ldc);

      auto gpuMatC = gpuPtrC
                         ? GPUArrayView2D<T>(gpuPtrC, n + cColStart, ldc, ldc)
                         : GPUArrayView2D<T>();

      tiles.emplace_back(comms[i], blasHandles[i], pinnedBuffers[2 * i],
                         pinnedBuffers[2 * i + 1], gpuBuffers[i * 3 + 2],
                         matrixDist, opA, alpha, matA, matB, beta, hostMatC,
                         gpuMatC);
    }

    std::vector<BlockInfo> blockInfos;
    blockInfos.reserve(descC.comm().size());

    IntType tileIdx = 0;

    // iterate grid wise
    for (IntType colStartIdx = 0; colStartIdx < matrixDist->num_block_cols();
         colStartIdx += descC.proc_grid_cols()) {
      for (IntType rowStartIdx = 0; rowStartIdx < matrixDist->num_block_rows();
           rowStartIdx += descC.proc_grid_rows()) {

        // iterate through blocks within grid
        for (IntType blockColIdx = colStartIdx;
             blockColIdx <
             std::min<IntType>(matrixDist->num_block_cols(),
                               colStartIdx + descC.proc_grid_cols());
             ++blockColIdx) {
          for (IntType blockRowIdx = rowStartIdx;
               blockRowIdx <
               std::min<IntType>(matrixDist->num_block_rows(),
                                 rowStartIdx + descC.proc_grid_rows());
               ++blockRowIdx) {

            blockInfos.emplace_back(
                matrixDist->get_block_info(blockRowIdx, blockColIdx));

            // Prepare processing when there are enough blocks to form ring
            if (blockInfos.size() == descC.comm().size()) {
              tiles[tileIdx % numTiles].prepare(blockInfos.begin(),
                                                blockInfos.end());
              blockInfos.resize(0);
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

    if (blockInfos.size()) {
      // Prepare with remaining blocks
      tiles[tileIdx].prepare(blockInfos.begin(), blockInfos.end());
      blockInfos.resize(0);
    }
    // Process remaining blocks
    bool tileToProcess = true;
    while (tileToProcess) {
      tileToProcess = false;
      for (auto &t : tiles) {
        tileToProcess |= t.process_step();
      }
    }


    // synchronize all streams
    for (auto &t : tiles) {
      t.synchronize();
    }


  } else {
    auto &pinnedBuffers = ctx.pinned_buffers(numTiles);
    const IntType numBlockRows = matrixDist->num_block_rows();
    const IntType numBlockCols = matrixDist->num_block_cols();

    const IntType numBlockRowsInTile =
        std::max<IntType>((ctx.tile_size_host() + descC.row_block_size() - 1) /
                              descC.row_block_size(),
                          1);
    const IntType numBlockColsInTile =
        std::max<IntType>((ctx.tile_size_host() + descC.col_block_size() - 1) /
                              descC.col_block_size(),
                          1);

    std::vector<AllReduceTileGPU<T>> tiles;
    tiles.reserve(numTiles);

    for (IntType i = 0; i < numTiles; ++i) {
      auto matA = gpuPtrA ? GPUMatrixAccessor<GPUArrayConstView2D<T>>(
                                GPUArrayConstView2D<T>(gpuPtrA, m, kLocal, lda))
                          : GPUMatrixAccessor<GPUArrayConstView2D<T>>(
                                HostArrayConstView2D<T>(A, m, kLocal, lda),
                                tileSizeGEMM, gpuBuffers[i * 3]);

      auto matB = gpuPtrB ? GPUMatrixAccessor<GPUArrayConstView2D<T>>(
                                GPUArrayConstView2D<T>(gpuPtrB, n, kLocal, ldb))
                          : GPUMatrixAccessor<GPUArrayConstView2D<T>>(
                                HostArrayConstView2D<T>(B, n, kLocal, ldb),
                                tileSizeGEMM, gpuBuffers[i * 3 + 1]);

      auto hostMatC = gpuPtrC ? HostArrayView2D<T>()
                              : HostArrayView2D<T>(C, n + cColStart, ldc, ldc);

      auto gpuMatC = gpuPtrC
                         ? GPUArrayView2D<T>(gpuPtrC, n + cColStart, ldc, ldc)
                         : GPUArrayView2D<T>();

      tiles.emplace_back(descC.comm(), blasHandles[i], pinnedBuffers[i],
                         gpuBuffers[i * 3 + 2], matrixDist, opA, alpha, matA,
                         matB, beta, hostMatC, gpuMatC, numBlockRowsInTile,
                         numBlockColsInTile);
    }

    if (ctx.num_threads() > 1) {
      // comm + worker thread
      SPLA_OMP_PRAGMA("omp parallel num_threads(2)") {
        GPUDeviceGuard deviceGuard(ctx.gpu_device_id());
        IntType counter = 0;
        for (IntType blockRowIdx = 0; blockRowIdx < numBlockRows;
             blockRowIdx += numBlockRowsInTile) {
          for (IntType blockColIdx = 0; blockColIdx < numBlockCols;
               blockColIdx += numBlockColsInTile, ++counter) {
            auto &t = tiles[counter % numTiles];
            if (omp_get_thread_num() == 0) {
              // wait for tile to be multiplied
              while (t.state() != TileState::Multiplied) {
              }
              t.exchange();
            } else {
              // wait for tile once encountering the same tile more than once
              if (counter >= numTiles) {
                while (t.state() != TileState::Exchanged) {
                }
                t.extract();
              }
              // start multiplication
              t.multiply(blockRowIdx, blockColIdx);
            }
          }
        }
      }
    } else {
      // single thread
      IntType counter = 0;
      for (IntType blockRowIdx = 0; blockRowIdx < numBlockRows;
           blockRowIdx += numBlockRowsInTile) {
        for (IntType blockColIdx = 0; blockColIdx < numBlockCols;
             blockColIdx += numBlockColsInTile, ++counter) {
          auto &t = tiles[counter % numTiles];
          if (t.state() == TileState::Multiplied) {
            t.exchange();
            t.extract();
          }
          t.multiply(blockRowIdx, blockColIdx);
        }
      }
    }

    // finalize remaining tiles
    for (auto &t : tiles) {
      if (t.state() == TileState::Multiplied) {
        t.exchange();
      }
      if (t.state() == TileState::Exchanged) {
        t.extract();
      }
    }

    // synchronize all streams
    for (auto &t : tiles) {
      t.synchronize();
    }
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
    int m, int n, int kLocal, SplaOperation opA, gpu::blas::ComplexFloatType alpha,
    const gpu::blas::ComplexFloatType *A, int lda, const gpu::blas::ComplexFloatType *B, int ldb,
    gpu::blas::ComplexFloatType beta, gpu::blas::ComplexFloatType *C, int ldc, int cRowStart,
    int cColStart, MatrixDistributionInternal &descC, ContextInternal &ctx);

template void pgemm_ssb_gpu<gpu::blas::ComplexDoubleType>(
    int m, int n, int kLocal, SplaOperation opA, gpu::blas::ComplexDoubleType alpha,
    const gpu::blas::ComplexDoubleType *A, int lda, const gpu::blas::ComplexDoubleType *B, int ldb,
    gpu::blas::ComplexDoubleType beta, gpu::blas::ComplexDoubleType *C, int ldc, int cRowStart,
    int cColStart, MatrixDistributionInternal &descC, ContextInternal &ctx);

}  // namespace spla
