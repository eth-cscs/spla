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
#ifndef SPLA_BLAS_THREADS_GUARD_HPP
#define SPLA_BLAS_THREADS_GUARD_HPP

#include "util/blas_interface.hpp"
#include "util/common_types.hpp"

namespace spla {
class BlasThreadsGuard {
public:
  explicit BlasThreadsGuard(IntType numThreadsTarget)
      : orignalNumThreads_(blas::get_num_threads()), numThreadsSet_(false) {
    if (orignalNumThreads_ != numThreadsTarget) {
      blas::set_num_threads(numThreadsTarget);
      numThreadsSet_ = true;
    }
  }

  BlasThreadsGuard() = delete;

  BlasThreadsGuard(const BlasThreadsGuard&) = delete;

  BlasThreadsGuard(BlasThreadsGuard&&) = delete;

  auto operator=(const BlasThreadsGuard&) -> BlasThreadsGuard& = delete;

  auto operator=(BlasThreadsGuard&&) -> BlasThreadsGuard& = delete;

  ~BlasThreadsGuard() {
    if (numThreadsSet_) blas::set_num_threads(orignalNumThreads_);
  }

private:
  IntType orignalNumThreads_;
  bool numThreadsSet_;
};
}  // namespace spla

#endif  // SPLA_BLAS_THREADS_GUARD_HPP
