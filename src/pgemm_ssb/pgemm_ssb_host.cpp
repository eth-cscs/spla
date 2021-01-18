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
#include <array>

#include "block_generation/block_cyclic_generator.hpp"
#include "block_generation/matrix_block_generator.hpp"
#include "block_generation/mirror_generator.hpp"
#include "gemm/gemm_host.hpp"
#include "mpi_util/mpi_check_status.hpp"
#include "spla/context_internal.hpp"
#include "spla/exceptions.hpp"
#include "spla/matrix_distribution_internal.hpp"
#include "spla/spla.hpp"
#include "spla/types.h"
#include "tile_host.hpp"
#include "timing/timing.hpp"
#include "util/blas_interface.hpp"
#include "util/blas_threads_guard.hpp"
#include "util/common_types.hpp"
#include "util/omp_definitions.hpp"
#include "util/check_gemm_param.hpp"
#include "mpi_util/mpi_match_elementary_type.hpp"
#include "mpi_util/mpi_request_handle.hpp"
#include "pgemm_ssb/ring_reduce_tile_host.hpp"


namespace spla {

template <typename T>
void pgemm_ssb_host(int m, int n, int kLocal, SplaOperation opA, T alpha, const T *A, int lda,
                    const T *B, int ldb, T beta, T *C, int ldc, int cRowStart, int cColStart,
                    MatrixDistributionInternal &descC, ContextInternal &ctx) {
  SCOPED_TIMING("inner_host");
  if (m == 0 || n == 0) {
    return;
  }
  if (opA != SplaOperation::SPLA_OP_TRANSPOSE && opA != SplaOperation::SPLA_OP_CONJ_TRANSPOSE) {
    throw InvalidParameterError();
  }

  if (m < 0 || n < 0 || cRowStart < 0 || cColStart < 0) {
    throw InvalidParameterError();
  }

  if (descC.comm().size() == 1) {
    return gemm_host<T>(ctx.num_threads(), opA, SPLA_OP_NONE, m, n, kLocal, alpha, A, lda, B, ldb,
                        beta, C + cRowStart + cColStart * ldc, ldc);
  }

  std::shared_ptr<MatrixBlockGenerator> matrixDist;
  if (descC.type() == SplaDistributionType::SPLA_DIST_BLACS_BLOCK_CYCLIC) {
    matrixDist.reset(new BlockCyclicGenerator(descC.row_block_size(), descC.col_block_size(),
                                              descC.proc_grid_rows(), descC.proc_grid_cols(), m, n,
                                              cRowStart, cColStart));
  } else {
    matrixDist.reset(new MirrorGenerator(ctx.tile_size_host(), ctx.tile_size_host(), m, n,
                                         cRowStart, cColStart));
  }

  check_gemm_param(opA, SplaOperation::SPLA_OP_NONE, matrixDist->local_rows(descC.comm().rank()),
                   matrixDist->local_cols(descC.comm().rank()), kLocal, A, lda, B, ldb, C, ldc);

  HostArrayConstView2D<T> viewA(A, m, kLocal, lda);
  HostArrayConstView2D<T> viewB(B, n, kLocal, ldb);
  HostArrayView2D<T> viewC(C, n + cColStart, ldc, ldc);

  auto buffers = ctx.mpi_buffers(2);

  std::array<RingReduceTileHost<T>, 2> tiles{
      RingReduceTileHost<T>{ctx.num_threads(), descC.comm(), buffers[0], matrixDist, opA, alpha,
       viewA, viewB, beta, viewC},
      RingReduceTileHost<T>{ctx.num_threads(), descC.comm(), buffers[1], matrixDist, opA, alpha,
       viewA, viewB, beta, viewC}};

  IntType tileIdx = 0;
  for (IntType blockColIdx = 0; blockColIdx < matrixDist->num_block_cols();
       blockColIdx += descC.proc_grid_cols()) {
    for (IntType blockRowIdx = 0; blockRowIdx < matrixDist->num_block_rows();
         blockRowIdx += descC.proc_grid_rows(), ++tileIdx) {
      const IntType numCurrentBlockRows =
          std::min<IntType>(matrixDist->num_block_rows() - blockRowIdx, descC.proc_grid_rows());
      const IntType numCurrentBlockCols =
          std::min<IntType>(matrixDist->num_block_cols() - blockColIdx, descC.proc_grid_cols());


      tiles[tileIdx % tiles.size()].prepare(blockRowIdx, blockColIdx, numCurrentBlockRows, numCurrentBlockCols);
      bool tileToProcess =true;

      while(tileToProcess) {
        tileToProcess = false;
        for(auto& t : tiles) {
          tileToProcess |= t.process_step();
        }
      }

    }
  }

}

template void pgemm_ssb_host<float>(int m, int n, int kLocal, SplaOperation opA, float alpha,
                                    const float *A, int lda, const float *B, int ldb, float beta,
                                    float *C, int ldc, int cRowStart, int cColStart,
                                    MatrixDistributionInternal &descC, ContextInternal &ctx);

template void pgemm_ssb_host<double>(int m, int n, int kLocal, SplaOperation opA, double alpha,
                                     const double *A, int lda, const double *B, int ldb,
                                     double beta, double *C, int ldc, int cRowStart, int cColStart,
                                     MatrixDistributionInternal &descC, ContextInternal &ctx);

template void pgemm_ssb_host<std::complex<float>>(
    int m, int n, int kLocal, SplaOperation opA, std::complex<float> alpha,
    const std::complex<float> *A, int lda, const std::complex<float> *B, int ldb,
    std::complex<float> beta, std::complex<float> *C, int ldc, int cRowStart, int cColStart,
    MatrixDistributionInternal &descC, ContextInternal &ctx);

template void pgemm_ssb_host<std::complex<double>>(
    int m, int n, int kLocal, SplaOperation opA, std::complex<double> alpha,
    const std::complex<double> *A, int lda, const std::complex<double> *B, int ldb,
    std::complex<double> beta, std::complex<double> *C, int ldc, int cRowStart, int cColStart,
    MatrixDistributionInternal &descC, ContextInternal &ctx);
}  // namespace spla
