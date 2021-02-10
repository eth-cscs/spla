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
#include "util/blas_interface.hpp"

#include <complex>

#include "spla/config.h"

// OpenBlas uses different types
#if defined(SPLA_BLAS_OPENBLAS)

using FloatComplexPtr = float *;
using DoubleComplexPtr = double *;
using ConstFloatComplexPtr = const float *;
using ConstDoubleComplexPtr = const double *;

#else

using FloatComplexPtr = void *;
using DoubleComplexPtr = void *;
using ConstFloatComplexPtr = const void *;
using ConstDoubleComplexPtr = const void *;

#endif

#ifdef SPLA_BLAS_BLIS
#include <blis.h>
#endif

// use blas header if found
#if defined(SPLA_BLAS_HEADER_NAME)

#include SPLA_BLAS_HEADER_NAME

#else

extern "C" {

enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 };

void cblas_sgemm(enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE transA, enum CBLAS_TRANSPOSE transB,
                 int M, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb,
                 float beta, float *C, int ldc);

void cblas_dgemm(enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE transA, enum CBLAS_TRANSPOSE transB,
                 int M, int N, int K, double alpha, const double *A, int lda, const double *B,
                 int ldb, double beta, double *C, int ldc);

void cblas_cgemm(enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE transA, enum CBLAS_TRANSPOSE transB,
                 int M, int N, int K, const void *alpha, const void *A, int lda, const void *B,
                 int ldb, const void *beta, void *C, int ldc);

void cblas_zgemm(enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE transA, enum CBLAS_TRANSPOSE transB,
                 int M, int N, int K, const void *alpha, const void *A, int lda, const void *B,
                 int ldb, const void *beta, void *C, int ldc);
}

#endif

namespace spla {
namespace blas {

static auto convert_operation(const Operation &op) -> CBLAS_TRANSPOSE {
  switch (op) {
    case Operation::TRANS:
      return CblasTrans;
    case Operation::CONJ_TRANS:
      return CblasConjTrans;
    default:
      return CblasNoTrans;
  }
}

auto gemm(Order order, Operation transA, Operation transB, IntType M, IntType N, IntType K,
          float alpha, const float *A, IntType lda, const float *B, IntType ldb, float beta,
          float *C, IntType ldc) -> void {
  CBLAS_ORDER cblasOrder = order == Order::COL_MAJOR ? CblasColMajor : CblasRowMajor;
  CBLAS_TRANSPOSE cblasTransA = convert_operation(transA);
  CBLAS_TRANSPOSE cblasTransB = convert_operation(transB);

  cblas_sgemm(cblasOrder, cblasTransA, cblasTransB, static_cast<int>(M), static_cast<int>(N),
              static_cast<int>(K), alpha, A, static_cast<int>(lda), B, static_cast<int>(ldb), beta,
              C, static_cast<int>(ldc));
}

auto gemm(Order order, Operation transA, Operation transB, IntType M, IntType N, IntType K,
          double alpha, const double *A, IntType lda, const double *B, IntType ldb, double beta,
          double *C, IntType ldc) -> void {
  CBLAS_ORDER cblasOrder = order == Order::COL_MAJOR ? CblasColMajor : CblasRowMajor;
  CBLAS_TRANSPOSE cblasTransA = convert_operation(transA);
  CBLAS_TRANSPOSE cblasTransB = convert_operation(transB);
  cblas_dgemm(cblasOrder, cblasTransA, cblasTransB, static_cast<int>(M), static_cast<int>(N),
              static_cast<int>(K), alpha, A, static_cast<int>(lda), B, static_cast<int>(ldb), beta,
              C, static_cast<int>(ldc));
}

auto gemm(Order order, Operation transA, Operation transB, IntType M, IntType N, IntType K,
          std::complex<float> alpha, const std::complex<float> *A, IntType lda,
          const std::complex<float> *B, IntType ldb, std::complex<float> beta,
          std::complex<float> *C, IntType ldc) -> void {
  CBLAS_ORDER cblasOrder = order == Order::COL_MAJOR ? CblasColMajor : CblasRowMajor;
  CBLAS_TRANSPOSE cblasTransA = convert_operation(transA);
  CBLAS_TRANSPOSE cblasTransB = convert_operation(transB);
  cblas_cgemm(cblasOrder, cblasTransA, cblasTransB, static_cast<int>(M), static_cast<int>(N),
              static_cast<int>(K), reinterpret_cast<ConstFloatComplexPtr>(&alpha),
              reinterpret_cast<ConstFloatComplexPtr>(A), static_cast<int>(lda),
              reinterpret_cast<ConstFloatComplexPtr>(B), static_cast<int>(ldb),
              reinterpret_cast<ConstFloatComplexPtr>(&beta), reinterpret_cast<FloatComplexPtr>(C),
              static_cast<int>(ldc));
}

auto gemm(Order order, Operation transA, Operation transB, IntType M, IntType N, IntType K,
          std::complex<double> alpha, const std::complex<double> *A, IntType lda,
          const std::complex<double> *B, IntType ldb, std::complex<double> beta,
          std::complex<double> *C, IntType ldc) -> void {
  CBLAS_ORDER cblasOrder = order == Order::COL_MAJOR ? CblasColMajor : CblasRowMajor;
  CBLAS_TRANSPOSE cblasTransA = convert_operation(transA);
  CBLAS_TRANSPOSE cblasTransB = convert_operation(transB);
  cblas_zgemm(cblasOrder, cblasTransA, cblasTransB, static_cast<int>(M), static_cast<int>(N),
              static_cast<int>(K), reinterpret_cast<ConstDoubleComplexPtr>(&alpha),
              reinterpret_cast<ConstDoubleComplexPtr>(A), static_cast<int>(lda),
              reinterpret_cast<ConstDoubleComplexPtr>(B), static_cast<int>(ldb),
              reinterpret_cast<ConstDoubleComplexPtr>(&beta), reinterpret_cast<DoubleComplexPtr>(C),
              static_cast<int>(ldc));
}

auto get_num_threads() -> IntType {
#if defined(SPLA_BLAS_OPENBLAS) && defined(SPLA_BLAS_HEADER_NAME)
  return openblas_get_num_threads();
#elif defined(SPLA_BLAS_MKL) && defined(SPLA_BLAS_HEADER_NAME)
  return mkl_get_max_threads();
#elif defined(SPLA_BLAS_BLIS)
  return bli_thread_get_num_threads();
#else
  return 1;
#endif
}

auto set_num_threads(IntType numThreads) -> void {
#if defined(SPLA_BLAS_OPENBLAS) && defined(SPLA_BLAS_HEADER_NAME)
  openblas_set_num_threads(numThreads);
#elif defined(SPLA_BLAS_MKL) && defined(SPLA_BLAS_HEADER_NAME)
  mkl_set_num_threads(numThreads);
#elif defined(SPLA_BLAS_BLIS)
  bli_thread_set_num_threads(numThreads);
#endif
}

auto is_parallel() -> bool {
#if defined(SPLA_BLAS_OPENBLAS) && defined(SPLA_BLAS_HEADER_NAME)
  return openblas_get_parallel();
#elif defined(SPLA_BLAS_MKL) && defined(SPLA_BLAS_HEADER_NAME)
  return mkl_get_max_threads() != 1;
#elif defined(SPLA_BLAS_BLIS)
  return bli_thread_get_num_threads() == 1;
#else
  return false;
#endif
}

auto is_thread_safe() -> bool {
#if defined(SPLA_BLAS_OPENBLAS)
  return false;
#else
  return true;
#endif
}

}  // namespace blas
}  // namespace spla
