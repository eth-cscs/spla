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
#include "gemm/gemm_host.hpp"
#include "memory/host_array_const_view.hpp"
#include "memory/host_array_view.hpp"
#include "mpi_util/mpi_check_status.hpp"
#include "pgemm_sbs/stripe_host.hpp"
#include "spla/context_internal.hpp"
#include "spla/exceptions.hpp"
#include "spla/matrix_distribution_internal.hpp"
#include "spla/spla.hpp"
#include "util/blas_interface.hpp"
#include "util/blas_threads_guard.hpp"
#include "util/common_types.hpp"
#include "util/omp_definitions.hpp"
#include "util/check_gemm_param.hpp"

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
void pgemm_sbs_host_internal(int mLocal, int n, int k, T alpha, const T *A,
                             int lda, const T *B, int ldb, int bRowOffset,
                             int bColOffset, MatrixDistributionInternal &descB,
                             T beta, T *C, int ldc, ContextInternal &ctx,
                             BLOCK_GEN gen) {

  check_gemm_param(SplaOperation::SPLA_OP_NONE, SplaOperation::SPLA_OP_NONE,
                   mLocal, gen.local_cols(descB.comm().rank()),
                   gen.local_rows(descB.comm().rank()), A, lda, B, ldb,
                   C, ldc);

  HostArrayConstView2D<T> viewA(A, k, mLocal, lda);
  HostArrayConstView2D<T> viewB(B, n + bColOffset, ldb, ldb);
  HostArrayView2D<T> viewC(C, n, mLocal, ldc);

  std::vector<StripeHost<T, BLOCK_GEN>> stripes;
  stripes.reserve(ctx.num_threads());

  const IntType numBlockCols = gen.num_block_cols();
  const IntType numBlockColsInTile = std::max<IntType>(
      (128 + descB.col_block_size() - 1) / descB.col_block_size(), 1);

  // create stripes
  {
    auto &buffers = ctx.mpi_buffers(2 * ctx.num_tiles());
    auto &comms = descB.get_comms(ctx.num_tiles());
    IntType idx = 0;
    for (IntType tileIdx = 0; tileIdx < ctx.num_tiles(); ++tileIdx, ++idx) {
      stripes.emplace_back(ctx.num_threads(), comms[idx], buffers[2 * idx],
                           buffers[2 * idx + 1], gen, alpha, viewA,
                           viewB, beta, viewC, numBlockColsInTile);
    }
  }

  IntType currentTileIdx = 0;
  for (IntType blockColIdx = 0; blockColIdx < numBlockCols;
       blockColIdx += numBlockColsInTile) {
    IntType nextTileIdx = (currentTileIdx + 1) % ctx.num_tiles();

    stripes[nextTileIdx].collect(blockColIdx);
    stripes[nextTileIdx].start_exchange();

    if (stripes[currentTileIdx].state() == StripeState::InExchange) {
      stripes[currentTileIdx].finalize_exchange();
      stripes[currentTileIdx].multiply();
    }

    currentTileIdx = nextTileIdx;
  }
  for (IntType i = 0; i < ctx.num_tiles(); ++i) {
    if (stripes[i].state() == StripeState::InExchange) {
      stripes[i].finalize_exchange();
    }
    if (stripes[i].state() == StripeState::Exchanged) {
      stripes[i].multiply();
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

  BlockCyclicGenerator gen(descB.row_block_size(), descB.col_block_size(),
                           descB.proc_grid_rows(), descB.proc_grid_cols(), k, n,
                           bRowOffset, bColOffset);
  pgemm_sbs_host_internal<T, BlockCyclicGenerator>(
      mLocal, n, k, alpha, A, lda, B, ldb, bRowOffset, bColOffset, descB, beta,
      C, ldc, ctx, gen);

  }

template void pgemm_sbs_host(int mLocal, int n, int k, float alpha,
                             const float *A, int lda, const float *B, int ldb,
                             int bRowOffset, int bColOffset,
                             MatrixDistributionInternal &descB, float beta,
                             float *C, int ldc, ContextInternal &ctx);

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
