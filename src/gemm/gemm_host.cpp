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

#include "gemm/gemm_host.hpp"

#include <algorithm>
#include <complex>

#include "memory/host_array_const_view.hpp"
#include "memory/host_array_view.hpp"
#include "spla/config.h"
#include "spla/context_internal.hpp"
#include "spla/types.h"
#include "util/blas_interface.hpp"
#include "util/blas_threads_guard.hpp"
#include "util/check_gemm_param.hpp"

namespace spla {

static auto map_op_to_host_blas(SplaOperation op) -> blas::Operation {
  switch (op) {
    case SplaOperation::SPLA_OP_TRANSPOSE:
      return blas::Operation::TRANS;
    case SplaOperation::SPLA_OP_CONJ_TRANSPOSE:
      return blas::Operation::CONJ_TRANS;
    default:
      return blas::Operation::NONE;
  }
}

template <typename T>
void gemm_host(IntType numThreads, SplaOperation opA, SplaOperation opB, IntType m, IntType n,
               IntType k, T alpha, const T *A, IntType lda, const T *B, IntType ldb, T beta, T *C,
               IntType ldc) {
  if (m == 0 || n == 0) {
    return;
  }
  check_gemm_param(opA, opB, m, n, k, A, lda, B, ldb, C, ldc);

  const auto opBlasA = map_op_to_host_blas(opA);
  const auto opBlasB = map_op_to_host_blas(opB);

  // Some blas libraries like MKL do not accept 0 as ld, even if m, n or k is 0
  if (lda < 1) lda = 1;
  if (ldb < 1) ldb = 1;
  if (ldc < 1) ldc = 1;

  const bool inParallel = omp_in_parallel();

  // if blas library is parallelized or not thread safe, call it directly
  if ((blas::is_parallel() && !inParallel) || !blas::is_thread_safe()) {
    SPLA_OMP_PRAGMA("omp master") {  // no implicit barrier
      BlasThreadsGuard threadGuard(numThreads);
      blas::gemm(blas::Order::COL_MAJOR, opBlasA, opBlasB, m, n, k, alpha, A, lda, B, ldb, beta, C,
                 ldc);
    }
    if (inParallel) {
      SPLA_OMP_PRAGMA("omp barrier")
    }
    return;
  }

  // assume blas is not parallelized
  HostArrayConstView2D<T> viewA(A, opA == SplaOperation::SPLA_OP_NONE ? k : m,
                                opA == SplaOperation::SPLA_OP_NONE ? m : k, lda);
  HostArrayConstView2D<T> viewB(B, opB == SplaOperation::SPLA_OP_NONE ? n : k,
                                opB == SplaOperation::SPLA_OP_NONE ? k : n, ldb);
  HostArrayView2D<T> viewC(C, n, m, ldc);

  // If there are multiple threads, use 2 times as many tiles to take advantage of dynamic
  // scheduling
  const IntType numThreadCols = numThreads;
  const IntType numThreadRows = numThreads > 1 ? 2 : 1;

  const IntType minBlockSize = 5;

  const IntType colBlockSize =
      std::min<IntType>((n + numThreadCols - 1) / numThreadCols, minBlockSize);
  const IntType rowBlockSize =
      std::min<IntType>((m + numThreadRows - 1) / numThreadRows, minBlockSize);

  if (omp_in_parallel()) {
    SPLA_OMP_PRAGMA("omp for schedule(dynamic) collapse(2)")
    for (IntType col = 0; col < n; col += colBlockSize) {
      for (IntType row = 0; row < m; row += rowBlockSize) {
        const IntType currentCols = std::min<IntType>(viewC.dim_outer() - col, colBlockSize);
        const IntType currentRows = std::min<IntType>(viewC.dim_inner() - row, rowBlockSize);
        const IntType rowA = opA == SplaOperation::SPLA_OP_NONE ? row : 0;
        const IntType colA = opA == SplaOperation::SPLA_OP_NONE ? 0 : row;
        const IntType rowB = opB == SplaOperation::SPLA_OP_NONE ? 0 : col;
        const IntType colB = opB == SplaOperation::SPLA_OP_NONE ? col : 0;
        blas::gemm(blas::Order::COL_MAJOR, opBlasA, opBlasB, currentRows, currentCols, k, alpha,
                   viewA.size() ? &viewA(colA, rowA) : nullptr, lda,
                   viewB.size() ? &viewB(colB, rowB) : nullptr, ldb, beta, &viewC(col, row), ldc);
      }
    }
  } else {
    SPLA_OMP_PRAGMA("omp parallel for schedule(dynamic) collapse(2) num_threads(numThreads)")
    for (IntType col = 0; col < n; col += colBlockSize) {
      for (IntType row = 0; row < m; row += rowBlockSize) {
        const IntType currentCols = std::min<IntType>(viewC.dim_outer() - col, colBlockSize);
        const IntType currentRows = std::min<IntType>(viewC.dim_inner() - row, rowBlockSize);
        const IntType rowA = opA == SplaOperation::SPLA_OP_NONE ? row : 0;
        const IntType colA = opA == SplaOperation::SPLA_OP_NONE ? 0 : row;
        const IntType rowB = opB == SplaOperation::SPLA_OP_NONE ? 0 : col;
        const IntType colB = opB == SplaOperation::SPLA_OP_NONE ? col : 0;
        blas::gemm(blas::Order::COL_MAJOR, opBlasA, opBlasB, currentRows, currentCols, k, alpha,
                   viewA.size() ? &viewA(colA, rowA) : nullptr, lda,
                   viewB.size() ? &viewB(colB, rowB) : nullptr, ldb, beta, &viewC(col, row), ldc);
      }
    }
  }
}

template auto gemm_host<float>(IntType numThreads, SplaOperation opA, SplaOperation opB, IntType m,
                               IntType n, IntType k, float alpha, const float *A, IntType lda,
                               const float *B, IntType ldb, float beta, float *C, IntType ldc)
    -> void;

template auto gemm_host<double>(IntType numThreads, SplaOperation opA, SplaOperation opB, IntType m,
                                IntType n, IntType k, double alpha, const double *A, IntType lda,
                                const double *B, IntType ldb, double beta, double *C, IntType ldc)
    -> void;

template auto gemm_host<std::complex<float>>(IntType numThreads, SplaOperation opA,
                                             SplaOperation opB, IntType m, IntType n, IntType k,
                                             std::complex<float> alpha,
                                             const std::complex<float> *A, IntType lda,
                                             const std::complex<float> *B, IntType ldb,
                                             std::complex<float> beta, std::complex<float> *C,
                                             IntType ldc) -> void;

template auto gemm_host<std::complex<double>>(IntType numThreads, SplaOperation opA,
                                              SplaOperation opB, IntType m, IntType n, IntType k,
                                              std::complex<double> alpha,
                                              const std::complex<double> *A, IntType lda,
                                              const std::complex<double> *B, IntType ldb,
                                              std::complex<double> beta, std::complex<double> *C,
                                              IntType ldc) -> void;

}  // namespace spla
