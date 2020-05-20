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
#include <atomic>
#include <memory>
#include <vector>

#include "block_generation/block_cyclic_generator.hpp"
#include "block_generation/matrix_block_generator.hpp"
#include "block_generation/mirror_generator.hpp"
#include "mpi_util/mpi_check_status.hpp"
#include "spla/context_internal.hpp"
#include "spla/exceptions.hpp"
#include "spla/matrix_distribution_internal.hpp"
#include "spla/spla.hpp"
#include "tile_host.hpp"
#include "timing/timing.hpp"
#include "util/blas_interface.hpp"
#include "util/blas_threads_guard.hpp"
#include "util/common_types.hpp"
#include "util/omp_definitions.hpp"
#include "gemm/gemm_host.hpp"

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
 *    ------       -
 *    |    |       |    |
 *    |    |       |    |
 *    ------       -
 *      A            B
 */
template <typename T>
void gemm_ssb_host(int m, int n, int kLocal, T alpha, const T *A, int lda, const T *B, int ldb,
                   T beta, T *C, int ldc, int cRowStart, int cColStart,
                   MatrixDistributionInternal &descC, ContextInternal &ctx) {

  if (m == 0 || n == 0) {
    return;
  }

  if (descC.comm().size() == 1) {
    return gemm_host<T>(SPLA_OP_CONJ_TRANSPOSE, SPLA_OP_NONE, m, n, kLocal,
                        alpha, A, lda, B, ldb, beta,
                        C + cRowStart + cColStart * ldc, ldc, ctx);
  }

  BlasThreadsGuard(1);  // make sure blas is not multithreaded. gemm_host() sets this internally

  HostArrayConstView2D<T> viewA(A, m, kLocal, lda);
  HostArrayConstView2D<T> viewB(B, n, kLocal, ldb);
  HostArrayView2D<T> viewC(C, n + cColStart, ldc, ldc);

  std::shared_ptr<MatrixBlockGenerator> matrixDist;
  if (descC.type() == SplaDistributionType::SPLA_DIST_BLACS_BLOCK_CYCLIC) {
    matrixDist.reset(new BlockCyclicGenerator(descC.row_block_size(), descC.col_block_size(),
                                              descC.proc_grid_rows(), descC.proc_grid_cols(), m, n,
                                              cRowStart, cColStart));
  } else {
    matrixDist.reset(new MirrorGenerator(ctx.tile_length_target(), ctx.tile_length_target(), m, n,
                                         cRowStart, cColStart));
  }
  const IntType numBlockRows = matrixDist->num_block_rows();
  const IntType numBlockCols = matrixDist->num_block_cols();

  const IntType numBlockRowsInTile =
      (ctx.tile_length_target() + matrixDist->max_rows_in_block() - 1) /
      matrixDist->max_rows_in_block();
  const IntType numBlockColsInTile =
      (ctx.tile_length_target() + matrixDist->max_cols_in_block() - 1) /
      matrixDist->max_cols_in_block();

  std::vector<std::vector<TileHost<T>>> threadTiles(ctx.num_threads());

  // create tiles
  {
    auto &buffers = ctx.mpi_buffers(ctx.num_threads() * ctx.num_tiles_per_thread());
    auto &comms = descC.get_comms(ctx.num_threads() * ctx.num_tiles_per_thread());
    IntType idx = 0;
    for (auto &tiles : threadTiles) {
      for (IntType tileIdx = 0; tileIdx < ctx.num_tiles_per_thread(); ++tileIdx, ++idx) {
        tiles.emplace_back(comms[idx], buffers[idx], matrixDist, alpha, viewA, viewB, beta, viewC,
                           numBlockRowsInTile, numBlockColsInTile);
      }
    }
  }

  // for (IntType i = 0; i < ctx.num_threads(); ++i) {
  //   threadTiles.emplace_back(ctx.get_thread_tile_buffers<T>(i), alpha, viewA, viewB, viewC,
  //   descC);
  // }

  const IntType numTilesPerThread = static_cast<IntType>(threadTiles[0].size());

  int mpiThreadSupport;
  mpi_check_status(MPI_Query_thread(&mpiThreadSupport));

  if (mpiThreadSupport == MPI_THREAD_MULTIPLE || ctx.num_threads() == 1) {
    SCOPED_TIMING("inner_host_thread_multiple");

    IntType currentTileIdx = 0;

    SPLA_OMP_PRAGMA(
        "omp parallel for schedule(static) collapse(2) num_threads(ctx.num_threads()) firstprivate(currentTileIdx)")
    for (IntType blockRowIdx = 0; blockRowIdx < numBlockRows; blockRowIdx += numBlockRowsInTile) {
      for (IntType blockColIdx = 0; blockColIdx < numBlockCols; blockColIdx += numBlockColsInTile) {
        auto &tiles = threadTiles[omp_get_thread_num()];

        IntType nextTileIdx = (currentTileIdx + 1) % numTilesPerThread;

        if (tiles[nextTileIdx].state() == TileState::InExchange) {
          if (omp_get_thread_num() == 0) START_TIMING("finalize_exchange");
          tiles[nextTileIdx].finalize_exchange();
          if (omp_get_thread_num() == 0) STOP_TIMING("finalize_exchange");
          if (omp_get_thread_num() == 0) START_TIMING("extract");
          tiles[nextTileIdx].extract();
          if (omp_get_thread_num() == 0) STOP_TIMING("extract");
        }

        if (omp_get_thread_num() == 0) START_TIMING("blas_multiply");
        tiles[currentTileIdx].multiply(blockRowIdx, blockColIdx);
        if (omp_get_thread_num() == 0) STOP_TIMING("blas_multiply");
        if (omp_get_thread_num() == 0) START_TIMING("start_exchange");
        tiles[currentTileIdx].start_exchange();
        if (omp_get_thread_num() == 0) STOP_TIMING("start_exchange");

        currentTileIdx = nextTileIdx;
      }
    }
    SPLA_OMP_PRAGMA("omp parallel num_threads(ctx.num_threads())") {
      auto &tiles = threadTiles[omp_get_thread_num()];
      const IntType numTiles = static_cast<IntType>(tiles.size());
      for (IntType i = 0; i < numTiles; ++i) {
        if (tiles[i].state() == TileState::InExchange) {
          if (omp_get_thread_num() == 0) START_TIMING("finalize_exchange");
          tiles[i].finalize_exchange();
          if (omp_get_thread_num() == 0) STOP_TIMING("finalize_exchange");
          if (omp_get_thread_num() == 0) START_TIMING("extract");
          tiles[i].extract();
          if (omp_get_thread_num() == 0) STOP_TIMING("extract");
        }
      }
    }
  } else {
    SCOPED_TIMING("inner_host_thread_funneled");
    /*
     * MPI not thread safe -> funnel through master thread
     */
    if (mpiThreadSupport < MPI_THREAD_FUNNELED) throw MPIError();

    std::atomic<int> numProcThreadsDone(0);
    std::vector<std::atomic<int>> procThreadDone(ctx.num_threads());
    for (auto &val : procThreadDone) {
      val.store(0, std::memory_order_relaxed);
    }

    const IntType numRowTiles = (numBlockRows + numBlockRowsInTile - 1) / numBlockRowsInTile;
    const IntType numColTiles = (numBlockCols + numBlockColsInTile - 1) / numBlockColsInTile;

    SPLA_OMP_PRAGMA("omp parallel num_threads(ctx.num_threads())") {
      const IntType numProcessingThreads = omp_get_num_threads() - 1;
      if (omp_get_thread_num() == 0) {
        /*
         * Master thread
         */

        // iterate over tiles as long as processing is active
        while (numProcThreadsDone.load(std::memory_order_relaxed) != numProcessingThreads) {
          for (IntType tileIdx = 0;
               tileIdx < numTilesPerThread &&
               numProcThreadsDone.load(std::memory_order_relaxed) != numProcessingThreads;
               ++tileIdx) {
            // check all threads
            for (IntType threadId = 0; threadId < numProcessingThreads; ++threadId) {
              auto &t = threadTiles[threadId][tileIdx];
              auto &tNext = threadTiles[threadId][(tileIdx + 1) % numTilesPerThread];

              // wait for tile to be multiplied
              while (!procThreadDone[threadId].load(std::memory_order_relaxed) &&
                     t.state() != TileState::Multiplied) {
                // if next tile is already in exchange and ready, finalize
                if (tNext.state() == TileState::InExchange &&
                    tNext.exchange_is_ready_and_active()) {
                  tNext.finalize_exchange();
                }
              }

              // if proc threads still active, tile must be in Multiplied state (due to while loop
              // before)
              if (!procThreadDone[threadId].load(std::memory_order_relaxed)) {
                t.start_exchange();
                if (tNext.state() == TileState::InExchange) {
                  tNext.finalize_exchange();
                }
              }
            }
          }
        }

        // start all remaining messages
        for (IntType i = 0; i < numTilesPerThread; ++i) {
          for (IntType threadId = 0; threadId < numProcessingThreads; ++threadId) {
            auto &t = threadTiles[threadId][i];
            const auto state = t.state();
            if (state == TileState::Multiplied) {
              t.start_exchange();
            }
          }
        }

        // receive all remaining messages
        for (IntType i = 0; i < numTilesPerThread; ++i) {
          for (IntType threadId = 0; threadId < numProcessingThreads; ++threadId) {
            auto &t = threadTiles[threadId][i];
            const auto state = t.state();
            if (state == TileState::InExchange) {
              t.finalize_exchange();
            }
          }
        }

      } else {
        /*
         * Processing threads
         */
        const IntType procThreadIdx = omp_get_thread_num() - 1;
        const IntType numIterPerThread =
            (numRowTiles * numColTiles + numProcessingThreads - 1) / numProcessingThreads;

        IntType currentTileIdx = 0;
        for (IntType i = procThreadIdx * numIterPerThread;
             i <
             std::min<IntType>(numRowTiles * numColTiles, (procThreadIdx + 1) * numIterPerThread);
             ++i) {
          const IntType blockRowIdx = (i % numRowTiles) * numBlockRowsInTile;
          const IntType blockColIdx = (i / numRowTiles) * numBlockColsInTile;
          auto &tiles = threadTiles[procThreadIdx];

          // wait for tile to be exchanged in master thread. Tile may be Empty.
          while (tiles[currentTileIdx].state() == TileState::Multiplied) {
          }
          while (tiles[currentTileIdx].state() == TileState::InExchange) {
          }
          // NOTE: combining into single while loop with "||" caused bug: state was sometimes
          // InExchange after exiting

          // extract if necessary
          if (tiles[currentTileIdx].state() == TileState::Exchanged) {
            tiles[currentTileIdx].extract();
          }

          tiles[currentTileIdx].multiply(blockRowIdx, blockColIdx);
          // tiles[currentTileIdx].multiply(tr, cRowStart, tc, cColStart);

          currentTileIdx = (currentTileIdx + 1) % numTilesPerThread;
        }

        procThreadDone[procThreadIdx].store(1, std::memory_order_relaxed);
        numProcThreadsDone.fetch_add(1, std::memory_order_relaxed);
      }

      // make sure communication is done
      SPLA_OMP_PRAGMA("omp barrier")

      // extract remaining tiles
      if (omp_get_thread_num() < numProcessingThreads) {
        for (IntType i = 0; i < numTilesPerThread; ++i) {
          auto &t = threadTiles[omp_get_thread_num()][i];
          const auto state = t.state();
          if (state == TileState::Exchanged) {
            t.extract();
          }
        }
      }
    }
  }
}

template void gemm_ssb_host<float>(int m, int n, int kLocal, float alpha, const float *A, int lda,
                                   const float *B, int ldb, float beta, float *C, int ldc,
                                   int cRowStart, int cColStart, MatrixDistributionInternal &descC,
                                   ContextInternal &ctx);

template void gemm_ssb_host<double>(int m, int n, int kLocal, double alpha, const double *A,
                                    int lda, const double *B, int ldb, double beta, double *C,
                                    int ldc, int cRowStart, int cColStart,
                                    MatrixDistributionInternal &descC, ContextInternal &ctx);

template void gemm_ssb_host<std::complex<float>>(
    int m, int n, int kLocal, std::complex<float> alpha, const std::complex<float> *A, int lda,
    const std::complex<float> *B, int ldb, std::complex<float> beta, std::complex<float> *C,
    int ldc, int cRowStart, int cColStart, MatrixDistributionInternal &descC, ContextInternal &ctx);

template void gemm_ssb_host<std::complex<double>>(
    int m, int n, int kLocal, std::complex<double> alpha, const std::complex<double> *A, int lda,
    const std::complex<double> *B, int ldb, std::complex<double> beta, std::complex<double> *C,
    int ldc, int cRowStart, int cColStart, MatrixDistributionInternal &descC, ContextInternal &ctx);
}  // namespace spla
