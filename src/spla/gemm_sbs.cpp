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

#include "spla/gemm_sbs.hpp"

#include <algorithm>
#include <atomic>
#include <memory>
#include <vector>

#include "gemm_sbs/gemm_sbs_host.hpp"
#include "spla/exceptions.hpp"

#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
#include "gemm_sbs/gemm_sbs_gpu.hpp"
#include "gpu_util/gpu_blas_api.hpp"
#endif

namespace spla {

void gemm_sbs(int mLocal, int n, int k, float alpha, const float *A, int lda, const float *B,
              int ldb, int bRowOffset, int bColOffset, MatrixDistribution &distB, float beta, float *C,
              int ldc, Context &ctx) {
  if (ctx.processing_unit() == SplaProcessingUnit::SPLA_PU_HOST) {
    gemm_sbs_host(mLocal, n, k, alpha, A, lda, B, ldb, bRowOffset, bColOffset,
                  *(distB.descInternal_), beta, C, ldc, *(ctx.ctxInternal_));
  } else {
#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
    gemm_sbs_gpu(mLocal, n, k, alpha, A, lda, B, ldb, bRowOffset, bColOffset,
                 *(distB.descInternal_), beta, C, ldc, *(ctx.ctxInternal_));
#else
    throw GPUSupportError();
#endif
  }
}

void gemm_sbs(int mLocal, int n, int k, double alpha, const double *A, int lda, const double *B,
              int ldb, int bRowOffset, int bColOffset, MatrixDistribution &distB, double beta,
              double *C, int ldc, Context &ctx) {
  if (ctx.processing_unit() == SplaProcessingUnit::SPLA_PU_HOST) {
    gemm_sbs_host(mLocal, n, k, alpha, A, lda, B, ldb, bRowOffset, bColOffset,
                  *(distB.descInternal_), beta, C, ldc, *(ctx.ctxInternal_));
  } else {
#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
    gemm_sbs_gpu(mLocal, n, k, alpha, A, lda, B, ldb, bRowOffset, bColOffset,
                 *(distB.descInternal_), beta, C, ldc, *(ctx.ctxInternal_));
#else
    throw GPUSupportError();
#endif
  }
}

void gemm_sbs(int mLocal, int n, int k, std::complex<float> alpha, const std::complex<float> *A,
              int lda, const std::complex<float> *B, int ldb, int bRowOffset, int bColOffset,
              MatrixDistribution &distB, std::complex<float> beta, std::complex<float> *C, int ldc,
              Context &ctx) {
  if (ctx.processing_unit() == SplaProcessingUnit::SPLA_PU_HOST) {
    gemm_sbs_host(mLocal, n, k, alpha, A, lda, B, ldb, bRowOffset, bColOffset,
                  *(distB.descInternal_), beta, C, ldc, *(ctx.ctxInternal_));
  } else {
#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
    gemm_sbs_gpu<gpu::blas::ComplexFloatType>(
        mLocal, n, k, gpu::blas::ComplexFloatType{alpha.real(), alpha.imag()},
        reinterpret_cast<const gpu::blas::ComplexFloatType *>(A), lda,
        reinterpret_cast<const gpu::blas::ComplexFloatType *>(B), ldb, bRowOffset, bColOffset,
        *(distB.descInternal_), gpu::blas::ComplexFloatType{beta.real(), beta.imag()},
        reinterpret_cast<gpu::blas::ComplexFloatType *>(C), ldc, *(ctx.ctxInternal_));
#else
    throw GPUSupportError();
#endif
  }
}

void gemm_sbs(int mLocal, int n, int k, std::complex<double> alpha, const std::complex<double> *A,
              int lda, const std::complex<double> *B, int ldb, int bRowOffset, int bColOffset,
              MatrixDistribution &distB, std::complex<double> beta, std::complex<double> *C, int ldc,
              Context &ctx) {
  if (ctx.processing_unit() == SplaProcessingUnit::SPLA_PU_HOST) {
    gemm_sbs_host(mLocal, n, k, alpha, A, lda, B, ldb, bRowOffset, bColOffset,
                  *(distB.descInternal_), beta, C, ldc, *(ctx.ctxInternal_));
  } else {
#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
    gemm_sbs_gpu<gpu::blas::ComplexDoubleType>(
        mLocal, n, k, gpu::blas::ComplexDoubleType{alpha.real(), alpha.imag()},
        reinterpret_cast<const gpu::blas::ComplexDoubleType *>(A), lda,
        reinterpret_cast<const gpu::blas::ComplexDoubleType *>(B), ldb, bRowOffset, bColOffset,
        *(distB.descInternal_), gpu::blas::ComplexDoubleType{beta.real(), beta.imag()},
        reinterpret_cast<gpu::blas::ComplexDoubleType *>(C), ldc, *(ctx.ctxInternal_));
#else
    throw GPUSupportError();
#endif
  }
}

}  // namespace spla
