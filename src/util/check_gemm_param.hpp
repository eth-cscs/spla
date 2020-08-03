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
#ifndef SPLA_CHECK_GEMM_PARAM_HPP
#define SPLA_CHECK_GEMM_PARAM_HPP

#include "spla/config.h"
#include "spla/exceptions.hpp"
#include "spla/types.h"

namespace spla {
inline auto check_gemm_param(SplaOperation opA, SplaOperation opB, int m, int n, int k,
                             const void* A, int lda, const void* B, int ldb, const void* C, int ldc)
    -> void {
  if (m < 0 || n < 0 || k < 0) throw InvalidParameterError();
  if ((opA == SplaOperation::SPLA_OP_NONE && lda < m) ||
      (opA != SplaOperation::SPLA_OP_NONE && lda < k))
    throw InvalidParameterError();
  if ((opB == SplaOperation::SPLA_OP_NONE && ldb < k) ||
      (opB != SplaOperation::SPLA_OP_NONE && ldb < n))
    throw InvalidParameterError();

  if ((k != 0 && m != 0) && !A) throw InvalidPointerError();
  if ((k != 0 && n != 0) && !B) throw InvalidPointerError();
  if ((n != 0 && m != 0) && !C) throw InvalidPointerError();
}

}  // namespace spla

#endif
