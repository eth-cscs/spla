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
#include <cmath>
#include <memory>
#include <vector>

#include "block_generation/block_cyclic_generator.hpp"
#include "block_generation/matrix_block_generator.hpp"
#include "block_generation/mirror_generator.hpp"
#include "gemm_sbs/stripe_host.hpp"
#include "memory/host_array_const_view.hpp"
#include "memory/host_array_view.hpp"
#include "mpi_util/mpi_check_status.hpp"
#include "spla/context_internal.hpp"
#include "spla/exceptions.hpp"
#include "spla/matrix_distribution_internal.hpp"
#include "spla/spla.hpp"
#include "util/blas_interface.hpp"
#include "util/blas_threads_guard.hpp"
#include "util/common_types.hpp"
#include "util/omp_definitions.hpp"

namespace spla {

template <typename T>
static void gemm_sbs_host_single_rank(int m, int n, int k, T alpha, const T *A, int lda, const T *B,
                                      int ldb, int bRowOffset, int bColOffset,
                                      MatrixDistributionInternal &descB, T beta, T *C, int ldc,
                                      ContextInternal &ctx) {
  HostArrayConstView2D<T> viewA(A, k, m, lda);
  HostArrayConstView2D<T> viewB(B + bRowOffset + bColOffset * ldb, n, k, ldb);
  HostArrayView2D<T> viewC(C, n, m, ldc);

  const IntType numThreadCols = static_cast<IntType>(std::sqrt(ctx.num_threads()));
  const IntType numThreadRows = (ctx.num_threads() + numThreadCols - 1) / numThreadCols;

  const IntType colBlockSize = (n + numThreadCols - 1) / numThreadCols;
  const IntType rowBlockSize = (m + numThreadRows - 1) / numThreadRows;

  SPLA_OMP_PRAGMA("omp parallel for schedule(static) collapse(2) num_threads(ctx.num_threads())")
  for (IntType col = 0; col < n; col += colBlockSize) {
    for (IntType row = 0; row < m; row += rowBlockSize) {
      const IntType currentCols = std::min<IntType>(viewC.dim_outer() - col, colBlockSize);
      const IntType currentRows = std::min<IntType>(viewC.dim_inner() - row, rowBlockSize);
      blas::gemm(blas::Order::COL_MAJOR, blas::Operation::NONE, blas::Operation::NONE, currentRows,
                 currentCols, k, alpha, &viewA(0, row), lda, &viewB(col, 0), ldb, beta,
                 &viewC(col, row), ldc);
    }
  }
}

/*
 *    ------                    ------
 *    |    |                    |    |
 *    |    |                    |    |
 *    ------      -------       ------
 *    |    |      |  |  |       |    |
 *    |    |   *  -------   =   |    |
 *    ------      |  |  |       ------
 *    |    |      -------       |    |
 *    |    |         C          |    |
 *    ------                    ------
 *    |    |                    |    |
 *    |    |                    |    |
 *    ------                    ------
 *      A                         B
 */
template <typename T>
void gemm_sbs_host(int mLocal, int n, int k, T alpha, const T *A, int lda, const T *B, int ldb,
                   int bRowOffset, int bColOffset, MatrixDistributionInternal &descB, T beta, T *C,
                   int ldc, ContextInternal &ctx) {
  BlasThreadsGuard(1);  // make sure blas is not multithreaded

  if (k == 0 || n == 0) {
    return;
  }

  if (descB.comm().size() == 1 || descB.type() == SplaDistributionType::SPLA_DIST_MIRROR) {
    return gemm_sbs_host_single_rank<T>(mLocal, n, k, alpha, A, lda, B, ldb, bRowOffset, bColOffset,
                                        descB, beta, C, ldc, ctx);
  }

  HostArrayConstView2D<T> viewA(A, k, mLocal, lda);
  HostArrayConstView2D<T> viewB(B, n + bColOffset, ldb, ldb);
  HostArrayView2D<T> viewC(C, n, mLocal, ldc);

  std::shared_ptr<MatrixBlockGenerator> matrixDist;
  if (descB.type() == SplaDistributionType::SPLA_DIST_BLACS_BLOCK_CYCLIC) {
    matrixDist.reset(new BlockCyclicGenerator(descB.row_block_size(), descB.col_block_size(),
                                              descB.proc_grid_rows(), descB.proc_grid_cols(), k, n,
                                              bRowOffset, bColOffset));
  } else {
    matrixDist.reset(new MirrorGenerator(ctx.tile_length_target(), ctx.tile_length_target(), k, n,
                                         bRowOffset, bColOffset));
  }

  std::vector<std::vector<StripeHost<T>>> threadTiles(ctx.num_threads());

  const IntType numBlockCols = matrixDist->num_block_cols();
  const IntType numBlockColsInTile =
      std::max<IntType>((128 + descB.col_block_size() - 1) / descB.col_block_size(), 1);

  // create tiles
  {
    auto &buffers = ctx.mpi_buffers(2 * ctx.num_threads() * ctx.num_tiles_per_thread());
    auto &comms = descB.get_comms(ctx.num_threads() * ctx.num_tiles_per_thread());
    IntType idx = 0;
    for (auto &tiles : threadTiles) {
      for (IntType tileIdx = 0; tileIdx < ctx.num_tiles_per_thread(); ++tileIdx, ++idx) {
        tiles.emplace_back(comms[idx], buffers[2 * idx], buffers[2 * idx + 1], matrixDist, alpha,
                           viewA, viewB, beta, viewC, numBlockColsInTile);
      }
    }
  }

  const IntType numTilesPerThread = static_cast<IntType>(threadTiles[0].size());

  int mpiThreadSupport;
  mpi_check_status(MPI_Query_thread(&mpiThreadSupport));

  if (mpiThreadSupport == MPI_THREAD_MULTIPLE || ctx.num_threads() == 1) {
    IntType currentTileIdx = 0;

    SPLA_OMP_PRAGMA(
        "omp parallel for schedule(static) num_threads(ctx.num_threads()) firstprivate(currentTileIdx)")
    for (IntType blockColIdx = 0; blockColIdx < numBlockCols; blockColIdx += numBlockColsInTile) {
      auto &tiles = threadTiles[omp_get_thread_num()];

      IntType nextTileIdx = (currentTileIdx + 1) % numTilesPerThread;

      if (tiles[nextTileIdx].state() == StripeState::InExchange) {
        tiles[nextTileIdx].finalize_exchange();
        tiles[nextTileIdx].multiply();
      }

      tiles[currentTileIdx].collect(blockColIdx);
      tiles[currentTileIdx].start_exchange();

      currentTileIdx = nextTileIdx;
    }
    SPLA_OMP_PRAGMA("omp parallel num_threads(ctx.num_threads())") {
      auto &tiles = threadTiles[omp_get_thread_num()];
      const IntType numTiles = static_cast<IntType>(tiles.size());
      for (IntType i = 0; i < numTiles; ++i) {
        if (tiles[i].state() == StripeState::InExchange) {
          tiles[i].finalize_exchange();
          tiles[i].multiply();
        }
      }
    }
  } else {
    /*
     * MPI not thread safe -> funnel through master thread
     */
    if (mpiThreadSupport < MPI_THREAD_FUNNELED) throw MPIThreadSupportError();

    std::atomic<int> numProcThreadsDone(0);
    std::vector<std::atomic<int>> procThreadDone(ctx.num_threads() - 1);
    for (auto &val : procThreadDone) {
      val.store(0, std::memory_order_relaxed);
    }

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
                     t.state() != StripeState::Collected) {
                // if next tile is already in exchange and ready, finalize
                if (tNext.state() == StripeState::InExchange &&
                    tNext.exchange_is_ready_and_active()) {
                  tNext.finalize_exchange();
                }
              }

              // if proc threads still active, tile must be in Multiplied state (due to while loop
              // before)
              if (!procThreadDone[threadId].load(std::memory_order_relaxed)) {
                t.start_exchange();
                if (tNext.state() == StripeState::InExchange) {
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
            if (state == StripeState::Collected) {
              t.start_exchange();
            }
          }
        }

        // receive all remaining messages
        for (IntType i = 0; i < numTilesPerThread; ++i) {
          for (IntType threadId = 0; threadId < numProcessingThreads; ++threadId) {
            auto &t = threadTiles[threadId][i];
            const auto state = t.state();
            if (state == StripeState::InExchange) {
              t.finalize_exchange();
            }
          }
        }

      } else {
        /*
         * Processing threads
         */
        const IntType procThreadIdx = omp_get_thread_num() - 1;
        IntType currentTileIdx = 0;

        const IntType numIter = (numBlockCols + numBlockColsInTile - 1) / numBlockColsInTile;
        const IntType numIterPerThread =
            (numIter + numProcessingThreads - 1) / numProcessingThreads;

        for (IntType blockColIdx = procThreadIdx * numIterPerThread * numBlockColsInTile;
             blockColIdx < std::min<IntType>(numBlockCols, (procThreadIdx + 1) * numIterPerThread *
                                                               numBlockColsInTile);
             blockColIdx += numBlockColsInTile) {
          auto &tiles = threadTiles[procThreadIdx];

          // wait for tile to be exchanged in master thread. Tile may be Empty.
          while (tiles[currentTileIdx].state() == StripeState::Collected) {
          }
          while (tiles[currentTileIdx].state() == StripeState::InExchange) {
          }
          // NOTE: combining into single while loop with "||" caused bug: state was sometimes
          // InExchange after exiting

          // extract if necessary
          if (tiles[currentTileIdx].state() == StripeState::Exchanged) {
            tiles[currentTileIdx].multiply();
          }

          tiles[currentTileIdx].collect(blockColIdx);
          // tiles[currentTileIdx].multiply(tr, cRowStart, tc, cColStart);

          currentTileIdx = (currentTileIdx + 1) % numTilesPerThread;
        }

        procThreadDone[procThreadIdx].store(1, std::memory_order_relaxed);
        numProcThreadsDone.fetch_add(1, std::memory_order_relaxed);
      }

      // make sure communication is done
      SPLA_OMP_PRAGMA("omp barrier")

      // multiply remaining tiles
      if (omp_get_thread_num() != 0 && omp_get_thread_num() < numProcessingThreads) {
        for (IntType i = 0; i < numTilesPerThread; ++i) {
          auto &t = threadTiles[omp_get_thread_num() - 1][i];
          const auto state = t.state();
          if (state == StripeState::Exchanged) {
            t.multiply();
          }
        }
      }
    }
  }
}

template void gemm_sbs_host(int mLocal, int n, int k, float alpha, const float *A, int lda,
                            const float *B, int ldb, int bRowOffset, int bColOffset,
                            MatrixDistributionInternal &descB, float beta, float *C, int ldc,
                            ContextInternal &ctx);

template void gemm_sbs_host(int mLocal, int n, int k, double alpha, const double *A, int lda,
                            const double *B, int ldb, int bRowOffset, int bColOffset,
                            MatrixDistributionInternal &descB, double beta, double *C, int ldc,
                            ContextInternal &ctx);

template void gemm_sbs_host(int mLocal, int n, int k, std::complex<float> alpha,
                            const std::complex<float> *A, int lda, const std::complex<float> *B,
                            int ldb, int bRowOffset, int bColOffset,
                            MatrixDistributionInternal &descB, std::complex<float> beta,
                            std::complex<float> *C, int ldc, ContextInternal &ctx);

template void gemm_sbs_host(int mLocal, int n, int k, std::complex<double> alpha,
                            const std::complex<double> *A, int lda, const std::complex<double> *B,
                            int ldb, int bRowOffset, int bColOffset,
                            MatrixDistributionInternal &descB, std::complex<double> beta,
                            std::complex<double> *C, int ldc, ContextInternal &ctx);

}  // namespace spla
