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

#include "spla/gemm.hpp"

#include "gemm/gemm_host.hpp"
#include "spla/exceptions.hpp"
#include "spla/gemm.h"

#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
#include "gemm/gemm_gpu.hpp"
#include "gpu_util/gpu_blas_api.hpp"
#endif

namespace spla {
void gemm(SplaOperation opA, SplaOperation opB, int m, int n, int k, float alpha, const float *A,
          int lda, const float *B, int ldb, float beta, float *C, int ldc, Context &ctx) {
  if (ctx.processing_unit() == SplaProcessingUnit::SPLA_PU_HOST) {
    gemm_host<float>(ctx.ctxInternal_->num_threads(), opA, opB, m, n, k, alpha, A, lda, B, ldb,
                     beta, C, ldc);
  } else {
#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
    gemm_gpu<float>(opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, *(ctx.ctxInternal_));
#else
    throw GPUSupportError();
#endif
  }
}

void gemm(SplaOperation opA, SplaOperation opB, int m, int n, int k, double alpha, const double *A,
          int lda, const double *B, int ldb, double beta, double *C, int ldc, Context &ctx) {
  if (ctx.processing_unit() == SplaProcessingUnit::SPLA_PU_HOST) {
    gemm_host<double>(ctx.ctxInternal_->num_threads(), opA, opB, m, n, k, alpha, A, lda, B, ldb,
                      beta, C, ldc);
  } else {
#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
    gemm_gpu<double>(opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, *(ctx.ctxInternal_));
#else
    throw GPUSupportError();
#endif
  }
}

void gemm(SplaOperation opA, SplaOperation opB, int m, int n, int k, std::complex<float> alpha,
          const std::complex<float> *A, int lda, const std::complex<float> *B, int ldb,
          std::complex<float> beta, std::complex<float> *C, int ldc, Context &ctx) {
  if (ctx.processing_unit() == SplaProcessingUnit::SPLA_PU_HOST) {
    gemm_host<std::complex<float>>(ctx.ctxInternal_->num_threads(), opA, opB, m, n, k, alpha, A,
                                   lda, B, ldb, beta, C, ldc);
  } else {
#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
    gemm_gpu<gpu::blas::ComplexFloatType>(
        opA, opB, m, n, k, gpu::blas::ComplexFloatType{alpha.real(), alpha.imag()},
        reinterpret_cast<const gpu::blas::ComplexFloatType *>(A), lda,
        reinterpret_cast<const gpu::blas::ComplexFloatType *>(B), ldb,
        gpu::blas::ComplexFloatType{beta.real(), beta.imag()},
        reinterpret_cast<gpu::blas::ComplexFloatType *>(C), ldc, *(ctx.ctxInternal_));
#else
    throw GPUSupportError();
#endif
  }
}

void gemm(SplaOperation opA, SplaOperation opB, int m, int n, int k, std::complex<double> alpha,
          const std::complex<double> *A, int lda, const std::complex<double> *B, int ldb,
          std::complex<double> beta, std::complex<double> *C, int ldc, Context &ctx) {
  if (ctx.processing_unit() == SplaProcessingUnit::SPLA_PU_HOST) {
    gemm_host<std::complex<double>>(ctx.ctxInternal_->num_threads(), opA, opB, m, n, k, alpha, A,
                                    lda, B, ldb, beta, C, ldc);
  } else {
#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
    gemm_gpu<gpu::blas::ComplexDoubleType>(
        opA, opB, m, n, k, gpu::blas::ComplexDoubleType{alpha.real(), alpha.imag()},
        reinterpret_cast<const gpu::blas::ComplexDoubleType *>(A), lda,
        reinterpret_cast<const gpu::blas::ComplexDoubleType *>(B), ldb,
        gpu::blas::ComplexDoubleType{beta.real(), beta.imag()},
        reinterpret_cast<gpu::blas::ComplexDoubleType *>(C), ldc, *(ctx.ctxInternal_));
#else
    throw GPUSupportError();
#endif
  }
}

}  // namespace spla

extern "C" {

SplaError spla_sgemm(SplaOperation opA, SplaOperation opB, int m, int n, int k, float alpha,
                     const float *A, int lda, const float *B, int ldb, float beta, float *C,
                     int ldc, SplaContext ctx) {
  try {
    spla::gemm(opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc,
               *reinterpret_cast<spla::Context *>(ctx));
  } catch (const spla::GenericError &e) {
    return e.error_code();
  } catch (...) {
    return SplaError::SPLA_UNKNOWN_ERROR;
  }
  return SplaError::SPLA_SUCCESS;
}

SplaError spla_dgemm(SplaOperation opA, SplaOperation opB, int m, int n, int k, double alpha,
                     const double *A, int lda, const double *B, int ldb, double beta, double *C,
                     int ldc, SplaContext ctx) {
  try {
    spla::gemm(opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc,
               *reinterpret_cast<spla::Context *>(ctx));
  } catch (const spla::GenericError &e) {
    return e.error_code();
  } catch (...) {
    return SplaError::SPLA_UNKNOWN_ERROR;
  }
  return SplaError::SPLA_SUCCESS;
}

SplaError spla_cgemm(SplaOperation opA, SplaOperation opB, int m, int n, int k, const void *alpha,
                     const void *A, int lda, const void *B, int ldb, const void *beta, void *C,
                     int ldc, SplaContext ctx) {
  try {
    spla::gemm(opA, opB, m, n, k, *reinterpret_cast<const std::complex<float> *>(alpha),
               reinterpret_cast<const std::complex<float> *>(A), lda,
               reinterpret_cast<const std::complex<float> *>(B), ldb,
               *reinterpret_cast<const std::complex<float> *>(beta),
               reinterpret_cast<std::complex<float> *>(C), ldc,
               *reinterpret_cast<spla::Context *>(ctx));
  } catch (const spla::GenericError &e) {
    return e.error_code();
  } catch (...) {
    return SplaError::SPLA_UNKNOWN_ERROR;
  }
  return SplaError::SPLA_SUCCESS;
}

SplaError spla_zgemm(SplaOperation opA, SplaOperation opB, int m, int n, int k, const void *alpha,
                     const void *A, int lda, const void *B, int ldb, const void *beta, void *C,
                     int ldc, SplaContext ctx) {
  try {
    spla::gemm(opA, opB, m, n, k, *reinterpret_cast<const std::complex<double> *>(alpha),
               reinterpret_cast<const std::complex<double> *>(A), lda,
               reinterpret_cast<const std::complex<double> *>(B), ldb,
               *reinterpret_cast<const std::complex<double> *>(beta),
               reinterpret_cast<std::complex<double> *>(C), ldc,
               *reinterpret_cast<spla::Context *>(ctx));
  } catch (const spla::GenericError &e) {
    return e.error_code();
  } catch (...) {
    return SplaError::SPLA_UNKNOWN_ERROR;
  }
  return SplaError::SPLA_SUCCESS;
}
}
