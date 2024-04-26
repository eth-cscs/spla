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

extern "C" {

#ifdef SPLA_CBLAS
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

#else

void sgemm_(const char* TRANSA, const char* TRANSB, const int* M, const int* N, const int* K,
            const void* ALPHA, const void* A, const int* LDA, const void* B, const int* LDB,
            const void* BETA, void* C, const int* LDC, int TRANSA_len, int TRANSB_len);

void dgemm_(const char* TRANSA, const char* TRANSB, const int* M, const int* N, const int* K,
            const void* ALPHA, const void* A, const int* LDA, const void* B, const int* LDB,
            const void* BETA, void* C, const int* LDC, int TRANSA_len, int TRANSB_len);

void cgemm_(const char* TRANSA, const char* TRANSB, const int* M, const int* N, const int* K,
            const void* ALPHA, const void* A, const int* LDA, const void* B, const int* LDB,
            const void* BETA, void* C, const int* LDC, int TRANSA_len, int TRANSB_len);

void zgemm_(const char* TRANSA, const char* TRANSB, const int* M, const int* N, const int* K,
            const void* ALPHA, const void* A, const int* LDA, const void* B, const int* LDB,
            const void* BETA, void* C, const int* LDC, int TRANSA_len, int TRANSB_len);
#endif
}

namespace spla {
namespace blas {

#ifdef SPLA_CBLAS
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
#else
static auto convert_operation(const Operation& op) -> const char* {
  switch (op) {
    case Operation::TRANS:
      return "T";
    case Operation::CONJ_TRANS:
      return "C";
    default:
      return "N";
  }
}
#endif

auto gemm(Operation transA, Operation transB, IntType M, IntType N, IntType K, float alpha,
          const float *A, IntType lda, const float *B, IntType ldb, float beta, float *C,
          IntType ldc) -> void {
#ifdef SPLA_CBLAS
  CBLAS_ORDER cblasOrder = CblasColMajor;
  CBLAS_TRANSPOSE cblasTransA = convert_operation(transA);
  CBLAS_TRANSPOSE cblasTransB = convert_operation(transB);

  cblas_sgemm(cblasOrder, cblasTransA, cblasTransB, static_cast<int>(M), static_cast<int>(N),
              static_cast<int>(K), alpha, A, static_cast<int>(lda), B, static_cast<int>(ldb), beta,
              C, static_cast<int>(ldc));
#else
  auto intM = static_cast<int>(M);
  auto intN = static_cast<int>(N);
  auto intK = static_cast<int>(K);
  auto intLda = static_cast<int>(lda);
  auto intLdb = static_cast<int>(ldb);
  auto intLdc = static_cast<int>(ldc);
  sgemm_(convert_operation(transA), convert_operation(transB), &intM, &intN, &intK, &alpha, A,
         &intLda, B, &intLdb, &beta, C, &intLdc, 1, 1);
#endif
}

auto gemm(Operation transA, Operation transB, IntType M, IntType N, IntType K, double alpha,
          const double *A, IntType lda, const double *B, IntType ldb, double beta, double *C,
          IntType ldc) -> void {
#ifdef SPLA_CBLAS
  CBLAS_ORDER cblasOrder = CblasColMajor;
  CBLAS_TRANSPOSE cblasTransA = convert_operation(transA);
  CBLAS_TRANSPOSE cblasTransB = convert_operation(transB);
  cblas_dgemm(cblasOrder, cblasTransA, cblasTransB, static_cast<int>(M), static_cast<int>(N),
              static_cast<int>(K), alpha, A, static_cast<int>(lda), B, static_cast<int>(ldb), beta,
              C, static_cast<int>(ldc));
#else
  auto intM = static_cast<int>(M);
  auto intN = static_cast<int>(N);
  auto intK = static_cast<int>(K);
  auto intLda = static_cast<int>(lda);
  auto intLdb = static_cast<int>(ldb);
  auto intLdc = static_cast<int>(ldc);
  dgemm_(convert_operation(transA), convert_operation(transB), &intM, &intN, &intK, &alpha, A,
         &intLda, B, &intLdb, &beta, C, &intLdc, 1, 1);
#endif
}

auto gemm(Operation transA, Operation transB, IntType M, IntType N, IntType K,
          std::complex<float> alpha, const std::complex<float> *A, IntType lda,
          const std::complex<float> *B, IntType ldb, std::complex<float> beta,
          std::complex<float> *C, IntType ldc) -> void {
#ifdef SPLA_CBLAS
  CBLAS_ORDER cblasOrder = CblasColMajor;
  CBLAS_TRANSPOSE cblasTransA = convert_operation(transA);
  CBLAS_TRANSPOSE cblasTransB = convert_operation(transB);
  cblas_cgemm(cblasOrder, cblasTransA, cblasTransB, static_cast<int>(M), static_cast<int>(N),
              static_cast<int>(K), &alpha, A, static_cast<int>(lda), B, static_cast<int>(ldb),
              &beta, C, static_cast<int>(ldc));
#else
  auto intM = static_cast<int>(M);
  auto intN = static_cast<int>(N);
  auto intK = static_cast<int>(K);
  auto intLda = static_cast<int>(lda);
  auto intLdb = static_cast<int>(ldb);
  auto intLdc = static_cast<int>(ldc);
  cgemm_(convert_operation(transA), convert_operation(transB), &intM, &intN, &intK, &alpha, A,
         &intLda, B, &intLdb, &beta, C, &intLdc, 1, 1);
#endif
}

auto gemm(Operation transA, Operation transB, IntType M, IntType N, IntType K,
          std::complex<double> alpha, const std::complex<double> *A, IntType lda,
          const std::complex<double> *B, IntType ldb, std::complex<double> beta,
          std::complex<double> *C, IntType ldc) -> void {
#ifdef SPLA_CBLAS
  CBLAS_ORDER cblasOrder = CblasColMajor;
  CBLAS_TRANSPOSE cblasTransA = convert_operation(transA);
  CBLAS_TRANSPOSE cblasTransB = convert_operation(transB);
  cblas_zgemm(cblasOrder, cblasTransA, cblasTransB, static_cast<int>(M), static_cast<int>(N),
              static_cast<int>(K), &alpha, A, static_cast<int>(lda), B, static_cast<int>(ldb),
              &beta, C, static_cast<int>(ldc));
#else
  auto intM = static_cast<int>(M);
  auto intN = static_cast<int>(N);
  auto intK = static_cast<int>(K);
  auto intLda = static_cast<int>(lda);
  auto intLdb = static_cast<int>(ldb);
  auto intLdc = static_cast<int>(ldc);
  zgemm_(convert_operation(transA), convert_operation(transB), &intM, &intN, &intK, &alpha, A,
         &intLda, B, &intLdb, &beta, C, &intLdc, 1, 1);
#endif
}

}  // namespace blas
}  // namespace spla
