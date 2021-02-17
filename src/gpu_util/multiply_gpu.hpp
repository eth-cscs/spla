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
#ifndef SPLA_MULTIPLY_GPU_HPP
#define SPLA_MULTIPLY_GPU_HPP

#include <cmath>
#include <vector>

#include "gpu_util/gpu_blas_api.hpp"
#include "gpu_util/gpu_blas_handle.hpp"
#include "gpu_util/gpu_helper.hpp"
#include "gpu_util/gpu_matrix_accessor.hpp"
#include "memory/gpu_array_const_view.hpp"
#include "memory/gpu_array_view.hpp"

namespace spla {
template <typename T>
auto multiply_gpu(const gpu::blas::HandleType &handle, gpu::blas::OperationType transa,
                  gpu::blas::OperationType transb, T alpha,
                  const GPUMatrixAccessor<GPUArrayConstView2D<T>> &tileA,
                  const GPUMatrixAccessor<GPUArrayConstView2D<T>> &tileB, T beta,
                  GPUArrayView2D<T> result) -> void;

}  // namespace spla
#endif
