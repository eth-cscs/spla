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

#ifndef SPLA_GPU_ARRAY_CONST_VIEW_HPP
#define SPLA_GPU_ARRAY_CONST_VIEW_HPP

#include <cassert>
#include <limits>

#include "memory/gpu_array_view.hpp"
#include "spla/config.h"
#include "spla/exceptions.hpp"
#include "util/common_types.hpp"

#if defined(__CUDACC__) || defined(__HIPCC__)
#include "gpu_util/gpu_runtime.hpp"
#endif

namespace spla {

template <typename T>
class GPUArrayConstView1D {
public:
  using ValueType = T;
  static constexpr IntType ORDER = 1;

  GPUArrayConstView1D() = default;

  GPUArrayConstView1D(const ValueType* data, const int size);

  GPUArrayConstView1D(const GPUArrayView1D<ValueType>& view);  // conversion allowed

#if defined(__CUDACC__) || defined(__HIPCC__)
  __device__ inline auto operator()(const int idx) -> ValueType {
    assert(idx < size_);
#if __CUDA_ARCH__ >= 350 || defined(__HIPCC__)
    return __ldg(data_ + idx);
#else
    return data_[idx];
#endif
  }

  __host__ __device__ inline auto data() const noexcept -> const ValueType* { return data_; }

  __host__ __device__ inline auto empty() const noexcept -> bool { return size_ == 0; }

  __host__ __device__ inline auto size() const noexcept -> int { return size_; }

  __host__ __device__ inline auto contiguous() const noexcept -> bool { return true; }

#else

  inline auto data() const noexcept -> const ValueType* { return data_; }

  inline auto empty() const noexcept -> bool { return size_ == 0; }

  inline auto size() const noexcept -> int { return size_; }

  inline auto contiguous() const noexcept -> bool { return true; }

#endif

private:
  int size_ = 0;
  const ValueType* data_ = nullptr;
};

template <typename T>
class GPUArrayConstView2D {
public:
  using ValueType = T;
  static constexpr IntType ORDER = 2;

  GPUArrayConstView2D() = default;

  GPUArrayConstView2D(const GPUArrayView2D<ValueType>& view);  // conversion allowed

  GPUArrayConstView2D(const ValueType* data, const int dimOuter, const int dimInner);

  GPUArrayConstView2D(const ValueType* data, const int dimOuter, const int dimInner,
                      const int ldInner);

#if defined(__CUDACC__) || defined(__HIPCC__)

  __device__ inline auto operator()(const int idxOuter, const int idxInner) -> ValueType {
    assert(idxOuter < dims_[0]);
    assert(idxInner < dims_[1]);
#if __CUDA_ARCH__ >= 350 || defined(__HIPCC__)
    return __ldg(data_ + idxOuter * ldInner_ + idxInner);
#else
    return data_[idxOuter * ldInner_ + idxInner];
#endif
  }

  __host__ __device__ inline auto index(const int idxOuter, const int idxInner) const noexcept
      -> int {
    return (idxOuter * ldInner_) + idxInner;
  }

  __host__ __device__ inline auto data() const noexcept -> const ValueType* { return data_; }

  __host__ __device__ inline auto empty() const noexcept -> bool { return this->size() == 0; }

  __host__ __device__ inline auto size() const noexcept -> int { return dims_[0] * dims_[1]; }

  __host__ __device__ inline auto dim_inner() const noexcept -> int { return dims_[1]; }

  __host__ __device__ inline auto dim_outer() const noexcept -> int { return dims_[0]; }

  __host__ __device__ inline auto ld_inner() const noexcept -> int { return ldInner_; }

  __host__ __device__ inline auto contiguous() const noexcept -> bool {
    return ldInner_ == dims_[1];
  }

#else

  inline auto index(const int idxOuter, const int idxInner) const noexcept -> int {
    return (idxOuter * ldInner_) + idxInner;
  }

  inline auto data() const noexcept -> const ValueType* { return data_; }

  inline auto empty() const noexcept -> bool { return this->size() == 0; }

  inline auto size() const noexcept -> int { return dims_[0] * dims_[1]; }

  inline auto dim_inner() const noexcept -> int { return dims_[1]; }

  inline auto dim_outer() const noexcept -> int { return dims_[0]; }

  inline auto ld_inner() const noexcept -> int { return ldInner_; }

  inline auto contiguous() const noexcept -> bool { return ldInner_ == dims_[1]; }

#endif

private:
  int dims_[2] = {0, 0};
  int ldInner_ = 0;
  const ValueType* data_ = nullptr;
};

template <typename T>
class GPUArrayConstView3D {
public:
  using ValueType = T;
  static constexpr IntType ORDER = 3;

  GPUArrayConstView3D() = default;

  GPUArrayConstView3D(const GPUArrayView3D<ValueType>& view);  // conversion allowed

  GPUArrayConstView3D(const ValueType* data, const int dimOuter, const int dimMid,
                      const int dimInner);

  GPUArrayConstView3D(const ValueType* data, const int dimOuter, const int dimMid,
                      const int dimInner, const int ldMid, const int ldInner);

#if defined(__CUDACC__) || defined(__HIPCC__)

  __device__ inline auto operator()(const int idxOuter, const int idxMid,
                                    const int idxInner) noexcept -> ValueType {
    assert(idxOuter < dims_[0]);
    assert(idxMid < dims_[1]);
    assert(idxInner < dims_[2]);
#if __CUDA_ARCH__ >= 350 || defined(__HIPCC__)
    return __ldg(data_ + (idxOuter * ldMid_ + idxMid) * ldInner_ + idxInner);
#else
    return data_[(idxOuter * ldMid_ + idxMid) * ldInner_ + idxInner];
#endif
  }

  __host__ __device__ inline auto index(const int idxOuter, const int idxMid,
                                        const int idxInner) const noexcept -> int {
    return (idxOuter * ldMid_ + idxMid) * ldInner_ + idxInner;
  }

  __host__ __device__ inline auto data() const noexcept -> const ValueType* { return data_; }

  __host__ __device__ inline auto empty() const noexcept -> bool { return this->size() == 0; }

  __host__ __device__ inline auto size() const noexcept -> int {
    return dims_[0] * dims_[1] * dims_[2];
  }

  __host__ __device__ inline auto dim_inner() const noexcept -> int { return dims_[2]; }

  __host__ __device__ inline auto dim_mid() const noexcept -> int { return dims_[1]; }

  __host__ __device__ inline auto dim_outer() const noexcept -> int { return dims_[0]; }

  __host__ __device__ inline auto ld_inner() const noexcept -> int { return ldInner_; }

  __host__ __device__ inline auto ld_mid() const noexcept -> int { return ldMid_; }

  __host__ __device__ inline auto contiguous() const noexcept -> bool {
    return ldInner_ == dims_[2] && ldMid_ == dims_[1];
  }

#else

  inline auto index(const int idxOuter, const int idxMid, const int idxInner) const noexcept
      -> int {
    return (idxOuter * ldMid_ + idxMid) * ldInner_ + idxInner;
  }

  inline auto data() const noexcept -> const ValueType* { return data_; }

  inline auto empty() const noexcept -> bool { return this->size() == 0; }

  inline auto size() const noexcept -> int { return dims_[0] * dims_[1] * dims_[2]; }

  inline auto dim_inner() const noexcept -> int { return dims_[2]; }

  inline auto dim_mid() const noexcept -> int { return dims_[1]; }

  inline auto dim_outer() const noexcept -> int { return dims_[0]; }

  inline auto ld_inner() const noexcept -> int { return ldInner_; }

  inline auto ld_mid() const noexcept -> int { return ldMid_; }

  inline auto contiguous() const noexcept -> bool {
    return ldInner_ == dims_[2] && ldMid_ == dims_[1];
  }

#endif

private:
  int dims_[3] = {0, 0, 0};
  int ldMid_ = 0;
  int ldInner_ = 0;
  const ValueType* data_ = nullptr;
};

// ======================
// Implementation
// ======================
template <typename T>
GPUArrayConstView1D<T>::GPUArrayConstView1D(const ValueType* data, const int size)
    : size_(size), data_(data) {
  assert(!(size != 0 && data == nullptr));
}

template <typename T>
GPUArrayConstView1D<T>::GPUArrayConstView1D(const GPUArrayView1D<ValueType>& view)
    : GPUArrayConstView1D<T>(view.data(), view.size()) {}

template <typename T>
GPUArrayConstView2D<T>::GPUArrayConstView2D(const ValueType* data, const int dimOuter,
                                            const int dimInner)
    : dims_{dimOuter, dimInner}, ldInner_(dimInner), data_(data) {
  assert(!(dimOuter != 0 && dimInner != 0 && data == nullptr));
  assert(dimOuter >= 0);
  assert(dimInner >= 0);
}

template <typename T>
GPUArrayConstView2D<T>::GPUArrayConstView2D(const ValueType* data, const int dimOuter,
                                            const int dimInner, const int ldInner)
    : dims_{dimOuter, dimInner}, ldInner_(ldInner), data_(data) {
  assert(!(dimOuter != 0 && dimInner != 0 && data == nullptr));
  assert(dimOuter >= 0);
  assert(dimInner >= 0);
  assert(ldInner >= dimInner);
}

template <typename T>
GPUArrayConstView2D<T>::GPUArrayConstView2D(const GPUArrayView2D<ValueType>& view)
    : GPUArrayConstView2D<T>(view.data(), view.dim_outer(), view.dim_inner(), view.ld_inner()) {}

template <typename T>
GPUArrayConstView3D<T>::GPUArrayConstView3D(const ValueType* data, const int dimOuter,
                                            const int dimMid, const int dimInner)
    : dims_{dimOuter, dimMid, dimInner}, ldMid_(dimMid), ldInner_(dimInner), data_(data) {
  assert(!(dimOuter != 0 && dimMid != 0 && dimInner != 0 && data == nullptr));
  assert(dimOuter >= 0);
  assert(dimMid >= 0);
  assert(dimInner >= 0);
}

template <typename T>
GPUArrayConstView3D<T>::GPUArrayConstView3D(const ValueType* data, const int dimOuter,
                                            const int dimMid, const int dimInner, const int ldMid,
                                            const int ldInner)
    : dims_{dimOuter, dimMid, dimInner}, ldMid_(ldMid), ldInner_(ldInner), data_(data) {
  assert(!(dimOuter != 0 && dimMid != 0 && dimInner != 0 && data == nullptr));
  assert(dimOuter >= 0);
  assert(dimMid >= 0);
  assert(dimInner >= 0);
  assert(ldMid >= dimMid);
  assert(ldInner >= dimInner);
}

template <typename T>
GPUArrayConstView3D<T>::GPUArrayConstView3D(const GPUArrayView3D<ValueType>& view)
    : GPUArrayConstView3D<T>(view.data(), view.dim_outer(), view.dim_mid(), view.dim_inner(),
                             view.ld_mid(), view.ld_inner()) {}

}  // namespace spla

#endif
