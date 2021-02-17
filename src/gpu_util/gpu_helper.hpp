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
#ifndef SPLA_GPU_HELPER_HPP
#define SPLA_GPU_HELPER_HPP

#include "spla/config.h"
#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
#include <complex>

#include "gpu_util/gpu_blas_api.hpp"
#include "gpu_util/gpu_runtime_api.hpp"

namespace spla {

template <typename T>
struct TypeTranslationHost;

template <>
struct TypeTranslationHost<float> {
  using type = float;
  static inline auto convert(const float& val) -> type { return val; }
};

template <>
struct TypeTranslationHost<double> {
  using type = double;
  static inline auto convert(const double& val) -> type { return val; }
};

template <>
struct TypeTranslationHost<std::complex<float>> {
  using type = std::complex<float>;
  static inline auto convert(const std::complex<float>& val) -> type { return val; }
};

template <>
struct TypeTranslationHost<std::complex<double>> {
  using type = std::complex<double>;
  static inline auto convert(const std::complex<double>& val) -> type { return val; }
};

template <>
struct TypeTranslationHost<gpu::blas::ComplexFloatType> {
  using type = std::complex<float>;
  static inline auto convert(const gpu::blas::ComplexFloatType& val) -> type {
    return type{val.x, val.y};
  }
};

template <>
struct TypeTranslationHost<gpu::blas::ComplexDoubleType> {
  using type = std::complex<double>;
  static inline auto convert(const gpu::blas::ComplexDoubleType& val) -> type {
    return type{val.x, val.y};
  }
};

template <typename T>
struct RealValueGPU;

template <>
struct RealValueGPU<float> {
  static inline auto create(float val) -> float { return val; }
};

template <>
struct RealValueGPU<double> {
  static inline auto create(double val) -> double { return val; }
};

template <>
struct RealValueGPU<gpu::blas::ComplexFloatType> {
  static inline auto create(float val) -> gpu::blas::ComplexFloatType { return {val, 0.f}; }
};

template <>
struct RealValueGPU<gpu::blas::ComplexDoubleType> {
  static inline auto create(double val) -> gpu::blas::ComplexDoubleType { return {val, 0.}; }
};

}  // namespace spla

#endif
#endif
