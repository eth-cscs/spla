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
#include <unordered_set>
#include <vector>

#include "block_generation/block.hpp"
#include "block_generation/block_cyclic_generator.hpp"
#include "block_generation/mirror_generator.hpp"
#include "gemm/gemm_host.hpp"
#include "memory/host_array_const_view.hpp"
#include "memory/host_array_view.hpp"
#include "mpi_util/mpi_check_status.hpp"
#include "pgemm_sbs/ring_sbs_host.hpp"
#include "spla/context_internal.hpp"
#include "spla/exceptions.hpp"
#include "spla/matrix_distribution_internal.hpp"
#include "spla/spla.hpp"
#include "util/blas_interface.hpp"
#include "util/blas_threads_guard.hpp"
#include "util/block_size_selection.hpp"
#include "util/check_gemm_param.hpp"
#include "util/common_types.hpp"
#include "util/omp_definitions.hpp"

namespace spla {

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
template <typename T, typename BLOCK_GEN>
void pgemm_sbs_host_internal(int mLocal, int n, int k, T alpha, const T *A, int lda, const T *B,
                             int ldb, int bRowOffset, int bColOffset,
                             MatrixDistributionInternal &descB, T beta, T *C, int ldc,
                             ContextInternal &ctx, BLOCK_GEN gen) {
  check_gemm_param(SplaOperation::SPLA_OP_NONE, SplaOperation::SPLA_OP_NONE, mLocal,
                   gen.local_cols(descB.comm().rank()), gen.local_rows(descB.comm().rank()), A, lda,
                   B, ldb, C, ldc);

  HostArrayConstView2D<T> viewA(A, k, mLocal, lda);
  HostArrayConstView2D<T> viewB(B, n + bColOffset, ldb, ldb);
  HostArrayView2D<T> viewC(C, n, mLocal, ldc);

  const double ringThreshold = 0.65;
  const IntType minBlockSize = 150;
  IntType rowsInBlock = 500;
  IntType colsInBlock = 500;

  std::tie(rowsInBlock, colsInBlock) = block_size_selection(
      SPLA_FILL_MODE_FULL, IsDisjointGenerator<BLOCK_GEN>::value, 1.0 - ringThreshold,
      descB.comm().size(), k, n, bRowOffset, bColOffset, ctx.tile_size_host(), minBlockSize);

  // Compute maximum block sizes such that memory allocations for increasing m / n can be avoided
  const IntType maxBlockSize =
      std::max<IntType>(rowsInBlock * colsInBlock, ctx.tile_size_host() * ctx.tile_size_host());

  constexpr IntType numTiles = 2;
  auto &buffers = ctx.mpi_buffers(numTiles);
  auto &comms = descB.get_comms(numTiles);

  std::array<RingSBSHost<T, BLOCK_GEN>, numTiles> tiles{
      RingSBSHost<T, BLOCK_GEN>{ringThreshold, maxBlockSize, ctx.num_threads(), comms[0],
                                buffers[0], gen, alpha, viewA, viewB, bRowOffset, bColOffset, beta,
                                viewC},
      RingSBSHost<T, BLOCK_GEN>{ringThreshold, maxBlockSize, ctx.num_threads(), comms[1],
                                buffers[1], gen, alpha, viewA, viewB, bRowOffset, bColOffset, beta,
                                viewC}};

  std::vector<Block> blocks;
  blocks.reserve(descB.comm().size());
  std::unordered_set<IntType> betaColIndeces;

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
            t.process_step(betaColIndeces);  // do one step for better comm / compute overlap
          }

          if (tileIdx == numTiles) {
            // All tiles are prepared -> start processing
            bool tileToProcess = true;
            while (tileToProcess) {
              tileToProcess = false;
              // Interleave processing to hide communication cost
              for (auto &t : tiles) {
                tileToProcess |= t.process_step(betaColIndeces);
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
    t.process_step(betaColIndeces);  // do one step for better comm / compute overlap
    blocks.resize(0);
  }
  // Process remaining blocks
  bool tileToProcess = true;
  while (tileToProcess) {
    tileToProcess = false;
    for (auto &t : tiles) {
      tileToProcess |= t.process_step(betaColIndeces);
    }
  }
}

template <typename T>
void pgemm_sbs_host(int mLocal, int n, int k, T alpha, const T *A, int lda, const T *B, int ldb,
                    int bRowOffset, int bColOffset, MatrixDistributionInternal &descB, T beta, T *C,
                    int ldc, ContextInternal &ctx) {
  if (k == 0 || n == 0) {
    return;
  }

  // Check if local operations only
  if (descB.comm().size() == 1 || descB.type() == SplaDistributionType::SPLA_DIST_MIRROR) {
    return gemm_host<T>(ctx.num_threads(), SPLA_OP_NONE, SPLA_OP_NONE, mLocal, n, k, alpha, A, lda,
                        B + bRowOffset + bColOffset * ldb, ldb, beta, C, ldc);
  }

  if (n < 0 || k < 0 || bRowOffset < 0 || bColOffset < 0) {
    throw InvalidParameterError();
  }

  BlockCyclicGenerator gen(descB.row_block_size(), descB.col_block_size(), descB.proc_grid_rows(),
                           descB.proc_grid_cols(), k, n, bRowOffset, bColOffset);
  pgemm_sbs_host_internal<T, BlockCyclicGenerator>(mLocal, n, k, alpha, A, lda, B, ldb, bRowOffset,
                                                   bColOffset, descB, beta, C, ldc, ctx, gen);
}

template void pgemm_sbs_host(int mLocal, int n, int k, float alpha, const float *A, int lda,
                             const float *B, int ldb, int bRowOffset, int bColOffset,
                             MatrixDistributionInternal &descB, float beta, float *C, int ldc,
                             ContextInternal &ctx);

template void pgemm_sbs_host(int mLocal, int n, int k, double alpha, const double *A, int lda,
                             const double *B, int ldb, int bRowOffset, int bColOffset,
                             MatrixDistributionInternal &descB, double beta, double *C, int ldc,
                             ContextInternal &ctx);

template void pgemm_sbs_host(int mLocal, int n, int k, std::complex<float> alpha,
                             const std::complex<float> *A, int lda, const std::complex<float> *B,
                             int ldb, int bRowOffset, int bColOffset,
                             MatrixDistributionInternal &descB, std::complex<float> beta,
                             std::complex<float> *C, int ldc, ContextInternal &ctx);

template void pgemm_sbs_host(int mLocal, int n, int k, std::complex<double> alpha,
                             const std::complex<double> *A, int lda, const std::complex<double> *B,
                             int ldb, int bRowOffset, int bColOffset,
                             MatrixDistributionInternal &descB, std::complex<double> beta,
                             std::complex<double> *C, int ldc, ContextInternal &ctx);

}  // namespace spla
