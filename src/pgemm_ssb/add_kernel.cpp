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

#include "pgemm_ssb/add_kernel.hpp"

#include <cassert>
#include <complex>
#include <cstring>

#include "spla/config.h"
#include "spla/types.h"

namespace spla {

template <typename T>
void add_kernel(IntType rows, IntType cols, const T *SPLA_RESTRICT_ATTR A, IntType lda, T beta,
                T *SPLA_RESTRICT_ATTR B, IntType ldb) {
  assert(lda >= rows);
  assert(ldb >= rows);
  if (beta == T(0.0)) {
    for (IntType c = 0; c < cols; ++c) {
      std::memcpy(B + c * ldb, A + c * lda, rows * sizeof(T));
    }
  } else if (beta == T(1.0)) {
    for (IntType c = 0; c < cols; ++c) {
      for (IntType r = 0; r < rows; ++r) {
        B[c * ldb + r] += A[c * lda + r];
      }
    }
  } else {
    for (IntType c = 0; c < cols; ++c) {
      for (IntType r = 0; r < rows; ++r) {
        B[c * ldb + r] = beta * B[c * ldb + r] + A[c * lda + r];
      }
    }
  }
}

template void add_kernel<float>(IntType rows, IntType cols, const float *A, IntType lda, float beta,
                                float *B, IntType ldb);

template void add_kernel<double>(IntType rows, IntType cols, const double *A, IntType lda,
                                 double beta, double *B, IntType ldb);

template void add_kernel<std::complex<float>>(IntType rows, IntType cols,
                                              const std::complex<float> *A, IntType lda,
                                              std::complex<float> beta, std::complex<float> *B,
                                              IntType ldb);

template void add_kernel<std::complex<double>>(IntType rows, IntType cols,
                                               const std::complex<double> *A, IntType lda,
                                               std::complex<double> beta, std::complex<double> *B,
                                               IntType ldb);

}  // namespace spla
