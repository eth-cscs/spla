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
#ifndef SPLA_BLAS_INTERFACE_HPP
#define SPLA_BLAS_INTERFACE_HPP

#include <complex>

#include "spla/config.h"
#include "util/common_types.hpp"

namespace spla {
namespace blas {

enum class Order { ROW_MAJOR = 101, COL_MAJOR = 102 };
enum class Operation { NONE = 111, TRANS = 112, CONJ_TRANS = 113 };

auto is_parallel() -> bool;

auto is_thread_safe() -> bool;

auto get_num_threads() -> IntType;

auto set_num_threads(IntType numThreads) -> void;

auto gemm(Order order, Operation transA, Operation transB, IntType M, IntType N, IntType K,
          float alpha, const float *A, IntType lda, const float *B, IntType ldb, float beta,
          float *C, IntType ldc) -> void;

auto gemm(Order order, Operation transA, Operation transB, IntType M, IntType N, IntType K,
          double alpha, const double *A, IntType lda, const double *B, IntType ldb, double beta,
          double *C, IntType ldc) -> void;

auto gemm(Order order, Operation transA, Operation transB, IntType M, IntType N, IntType K,
          std::complex<float> alpha, const std::complex<float> *A, IntType lda,
          const std::complex<float> *B, IntType ldb, std::complex<float> beta,
          std::complex<float> *C, IntType ldc) -> void;

auto gemm(Order order, Operation transA, Operation transB, IntType M, IntType N, IntType K,
          std::complex<double> alpha, const std::complex<double> *A, IntType lda,
          const std::complex<double> *B, IntType ldb, std::complex<double> beta,
          std::complex<double> *C, IntType ldc) -> void;

}  // namespace blas
}  // namespace spla

#endif
