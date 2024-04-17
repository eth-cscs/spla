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

#include "block_generation/block.hpp"
#include "block_generation/block_cyclic_generator.hpp"
#include "block_generation/mirror_generator.hpp"
#include "gemm/gemm_host.hpp"
#include "mpi_util/mpi_check_status.hpp"
#include "pgemm_ssb/ring_ssb_host.hpp"
#include "spla/context_internal.hpp"
#include "spla/exceptions.hpp"
#include "spla/matrix_distribution_internal.hpp"
#include "spla/spla.hpp"
#include "spla/types.h"
#include "timing/timing.hpp"
#include "util/blas_interface.hpp"
#include "util/block_size_selection.hpp"
#include "util/check_gemm_param.hpp"
#include "util/common_types.hpp"

namespace spla {

template <typename T, typename BLOCK_GEN>
void pgemm_ssb_host_internal(int m, int n, int kLocal, SplaOperation opA, T alpha, const T *A,
                             int lda, const T *B, int ldb, T beta, T *C, int ldc, int cRowOffset,
                             int cColOffset, SplaFillMode cFillMode,
                             MatrixDistributionInternal &descC, ContextInternal &ctx,
                             BLOCK_GEN gen) {
  SCOPED_TIMING("pgemm_ssb_host")
  check_gemm_param(opA, SplaOperation::SPLA_OP_NONE, gen.local_rows(descC.comm().rank()),
                   gen.local_cols(descC.comm().rank()), kLocal, A, lda, B, ldb, C, ldc);

  constexpr IntType numTiles = 2;

  IntType rowsInBlock = 1;
  IntType colsInBlock = 1;

  const double ringThreshold = 0.65;
  const IntType minBlockSize = 150;

  std::tie(rowsInBlock, colsInBlock) = block_size_selection(
      cFillMode, IsDisjointGenerator<BLOCK_GEN>::value, 1.0 - ringThreshold, descC.comm().size(), m,
      n, cRowOffset, cColOffset, ctx.tile_size_host(), minBlockSize);

  // Compute maximum block sizes such that memory allocations for increasing m / n can be avoided
  const IntType maxBlockSize =
      std::max<IntType>(rowsInBlock * colsInBlock, ctx.tile_size_host() * ctx.tile_size_host());

  HostArrayConstView2D<T> viewA(A, m, kLocal, lda);
  HostArrayConstView2D<T> viewB(B, n, kLocal, ldb);
  HostArrayView2D<T> viewC(C, n + cColOffset, ldc, ldc);

  auto &comms = descC.get_comms(numTiles);

  std::array<RingSSBHost<T, BLOCK_GEN>, numTiles> tiles{
      RingSSBHost<T, BLOCK_GEN>{ringThreshold, maxBlockSize, comms[0], ctx.allocators().host(), gen,
                                opA, alpha, viewA, viewB, beta, viewC},
      RingSSBHost<T, BLOCK_GEN>{ringThreshold, maxBlockSize, comms[1], ctx.allocators().host(), gen,
                                opA, alpha, viewA, viewB, beta, viewC}};

  std::vector<Block> blocks;
  blocks.reserve(descC.comm().size());

  IntType tileIdx = 0;

  // iterate grid wise
  for (IntType colStartIdx = 0; colStartIdx < n;
       colStartIdx += descC.proc_grid_cols() * colsInBlock) {
    for (IntType rowStartIdx = 0; rowStartIdx < m;
         rowStartIdx += descC.proc_grid_rows() * rowsInBlock) {
      // iterate through blocks within grid
      for (IntType colIdx = colStartIdx;
           colIdx < std::min<IntType>(n, colStartIdx + descC.proc_grid_cols() * colsInBlock);
           colIdx += colsInBlock) {
        for (IntType rowIdx = rowStartIdx;
             rowIdx < std::min<IntType>(m, rowStartIdx + descC.proc_grid_rows() * rowsInBlock);
             rowIdx += rowsInBlock) {
          const auto block = Block{rowIdx, colIdx, std::min<IntType>(rowsInBlock, m - rowIdx),
                                   std::min<IntType>(colsInBlock, n - colIdx)};
          if (block_is_active(block, cRowOffset, cColOffset, cFillMode)) blocks.emplace_back(block);

          // Prepare processing when there are enough blocks to form ring
          if (blocks.size() == descC.comm().size()) {
            tiles[tileIdx % numTiles].prepare(blocks.begin(), blocks.end());
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
}

template <typename T>
void pgemm_ssb_host(int m, int n, int kLocal, SplaOperation opA, T alpha, const T *A, int lda,
                    const T *B, int ldb, T beta, T *C, int ldc, int cRowOffset, int cColOffset,
                    SplaFillMode cFillMode, MatrixDistributionInternal &descC,
                    ContextInternal &ctx) {
  SCOPED_TIMING("inner_host");
  if (m == 0 || n == 0) {
    return;
  }
  if (opA != SplaOperation::SPLA_OP_TRANSPOSE && opA != SplaOperation::SPLA_OP_CONJ_TRANSPOSE) {
    throw InvalidParameterError();
  }

  if (m < 0 || n < 0 || cRowOffset < 0 || cColOffset < 0) {
    throw InvalidParameterError();
  }

  if (descC.comm().size() == 1) {
    return gemm_host<T>(opA, SPLA_OP_NONE, m, n, kLocal, alpha, A, lda, B, ldb, beta,
                        C + cRowOffset + cColOffset * ldc, ldc);
  }

  if (descC.type() == SplaDistributionType::SPLA_DIST_BLACS_BLOCK_CYCLIC) {
    BlockCyclicGenerator gen(descC.row_block_size(), descC.col_block_size(), descC.proc_grid_rows(),
                             descC.proc_grid_cols(), m, n, cRowOffset, cColOffset);

    pgemm_ssb_host_internal<T, BlockCyclicGenerator>(m, n, kLocal, opA, alpha, A, lda, B, ldb, beta,
                                                     C, ldc, cRowOffset, cColOffset, cFillMode,
                                                     descC, ctx, std::move(gen));

  } else {
    MirrorGenerator gen(ctx.tile_size_host(), ctx.tile_size_host(), m, n, cRowOffset, cColOffset);
    pgemm_ssb_host_internal<T, MirrorGenerator>(m, n, kLocal, opA, alpha, A, lda, B, ldb, beta, C,
                                                ldc, cRowOffset, cColOffset, cFillMode, descC, ctx,
                                                std::move(gen));
  }
}

template void pgemm_ssb_host<float>(int m, int n, int kLocal, SplaOperation opA, float alpha,
                                    const float *A, int lda, const float *B, int ldb, float beta,
                                    float *C, int ldc, int cRowOffset, int cColOffset,
                                    SplaFillMode cFillMode, MatrixDistributionInternal &descC,
                                    ContextInternal &ctx);

template void pgemm_ssb_host<double>(int m, int n, int kLocal, SplaOperation opA, double alpha,
                                     const double *A, int lda, const double *B, int ldb,
                                     double beta, double *C, int ldc, int cRowOffset,
                                     int cColOffset, SplaFillMode cFillMode,
                                     MatrixDistributionInternal &descC, ContextInternal &ctx);

template void pgemm_ssb_host<std::complex<float>>(
    int m, int n, int kLocal, SplaOperation opA, std::complex<float> alpha,
    const std::complex<float> *A, int lda, const std::complex<float> *B, int ldb,
    std::complex<float> beta, std::complex<float> *C, int ldc, int cRowOffset, int cColOffset,
    SplaFillMode cFillMode, MatrixDistributionInternal &descC, ContextInternal &ctx);

template void pgemm_ssb_host<std::complex<double>>(
    int m, int n, int kLocal, SplaOperation opA, std::complex<double> alpha,
    const std::complex<double> *A, int lda, const std::complex<double> *B, int ldb,
    std::complex<double> beta, std::complex<double> *C, int ldc, int cRowOffset, int cColOffset,
    SplaFillMode cFillMode, MatrixDistributionInternal &descC, ContextInternal &ctx);
}  // namespace spla
