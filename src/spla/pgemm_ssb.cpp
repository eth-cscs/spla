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

#include "spla/pgemm_ssb.hpp"

#include <algorithm>
#include <atomic>
#include <memory>
#include <vector>

#include "pgemm_ssb/pgemm_ssb_host.hpp"
#include "spla/exceptions.hpp"
#include "spla/pgemm_ssb.h"
#include "spla/pgemm_ssbtr.h"
#include "spla/pgemm_ssbtr.hpp"

#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
#include "gpu_util/gpu_blas_api.hpp"
#include "pgemm_ssb/pgemm_ssb_gpu.hpp"
#endif

namespace spla {
void pgemm_ssb(int m, int n, int kLocal, SplaOperation opA, float alpha, const float *A, int lda,
               const float *B, int ldb, float beta, float *C, int ldc, int cRowOffset,
               int cColOffset, MatrixDistribution &distC, Context &ctx) {
  pgemm_ssbtr(m, n, kLocal, opA, alpha, A, lda, B, ldb, beta, C, ldc, cRowOffset, cColOffset,
              SPLA_FILL_MODE_FULL, distC, ctx);
}

void pgemm_ssb(int m, int n, int kLocal, SplaOperation opA, double alpha, const double *A, int lda,
               const double *B, int ldb, double beta, double *C, int ldc, int cRowOffset,
               int cColOffset, MatrixDistribution &distC, Context &ctx) {
  pgemm_ssbtr(m, n, kLocal, opA, alpha, A, lda, B, ldb, beta, C, ldc, cRowOffset, cColOffset,
              SPLA_FILL_MODE_FULL, distC, ctx);
}

void pgemm_ssb(int m, int n, int kLocal, SplaOperation opA, std::complex<float> alpha,
               const std::complex<float> *A, int lda, const std::complex<float> *B, int ldb,
               std::complex<float> beta, std::complex<float> *C, int ldc, int cRowOffset,
               int cColOffset, MatrixDistribution &distC, Context &ctx) {
  pgemm_ssbtr(m, n, kLocal, opA, alpha, A, lda, B, ldb, beta, C, ldc, cRowOffset, cColOffset,
              SPLA_FILL_MODE_FULL, distC, ctx);
}

void pgemm_ssb(int m, int n, int kLocal, SplaOperation opA, std::complex<double> alpha,
               const std::complex<double> *A, int lda, const std::complex<double> *B, int ldb,
               std::complex<double> beta, std::complex<double> *C, int ldc, int cRowOffset,
               int cColOffset, MatrixDistribution &distC, Context &ctx) {
  pgemm_ssbtr(m, n, kLocal, opA, alpha, A, lda, B, ldb, beta, C, ldc, cRowOffset, cColOffset,
              SPLA_FILL_MODE_FULL, distC, ctx);
}
}  // namespace spla

extern "C" {

SplaError spla_psgemm_ssb(int m, int n, int kLocal, SplaOperation opA, float alpha, const float *A,
                          int lda, const float *B, int ldb, float beta, float *C, int ldc,
                          int cRowOffset, int cColOffset, SplaMatrixDistribution distC,
                          SplaContext ctx) {
  return spla_psgemm_ssbtr(m, n, kLocal, opA, alpha, A, lda, B, ldb, beta, C, ldc, cRowOffset,
                           cColOffset, SPLA_FILL_MODE_FULL, distC, ctx);
}

SplaError spla_pdgemm_ssb(int m, int n, int kLocal, SplaOperation opA, double alpha,
                          const double *A, int lda, const double *B, int ldb, double beta,
                          double *C, int ldc, int cRowOffset, int cColOffset,
                          SplaMatrixDistribution distC, SplaContext ctx) {
  return spla_pdgemm_ssbtr(m, n, kLocal, opA, alpha, A, lda, B, ldb, beta, C, ldc, cRowOffset,
                           cColOffset, SPLA_FILL_MODE_FULL, distC, ctx);
}

SplaError spla_pcgemm_ssb(int m, int n, int kLocal, SplaOperation opA, const void *alpha,
                          const void *A, int lda, const void *B, int ldb, const void *beta, void *C,
                          int ldc, int cRowOffset, int cColOffset, SplaMatrixDistribution distC,
                          SplaContext ctx) {
  return spla_pcgemm_ssbtr(m, n, kLocal, opA, alpha, A, lda, B, ldb, beta, C, ldc, cRowOffset,
                           cColOffset, SPLA_FILL_MODE_FULL, distC, ctx);
}

SplaError spla_pzgemm_ssb(int m, int n, int kLocal, SplaOperation opA, const void *alpha,
                          const void *A, int lda, const void *B, int ldb, const void *beta, void *C,
                          int ldc, int cRowOffset, int cColOffset, SplaMatrixDistribution distC,
                          SplaContext ctx) {
  return spla_pzgemm_ssbtr(m, n, kLocal, opA, alpha, A, lda, B, ldb, beta, C, ldc, cRowOffset,
                           cColOffset, SPLA_FILL_MODE_FULL, distC, ctx);
}
}
