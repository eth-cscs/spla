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

#ifndef SPLA_HOST_ARRAY_VIEW_HPP
#define SPLA_HOST_ARRAY_VIEW_HPP

#include <array>
#include <cassert>

#include "spla/config.h"
#include "util/common_types.hpp"

namespace spla {

template <typename T>
class HostArrayView1D {
public:
  using ValueType = T;
  using Iterator = T*;
  using ConstIterator = const T*;

  static constexpr IntType ORDER = 1;

  HostArrayView1D() = default;

  HostArrayView1D(ValueType* data, const IntType size);

  inline auto operator()(const IntType idx) -> ValueType& {
    assert(idx < size_);
    assert(idx >= 0);
    return data_[idx];
  }

  inline auto operator()(const IntType idx) const -> const ValueType& {
    assert(idx < size_);
    assert(idx >= 0);
    return data_[idx];
  }

  inline auto data() noexcept -> ValueType* { return data_; }

  inline auto data() const noexcept -> const ValueType* { return data_; }

  inline auto empty() const noexcept -> bool { return size_ == 0; }

  inline auto size() const noexcept -> IntType { return size_; }

  inline auto begin() noexcept -> Iterator { return data_; }

  inline auto begin() const noexcept -> ConstIterator { return data_; }

  inline auto cbegin() const noexcept -> ConstIterator { return data_; }

  inline auto end() noexcept -> Iterator { return data_ + size_; }

  inline auto end() const noexcept -> ConstIterator { return data_ + size_; }

  inline auto cend() const noexcept -> ConstIterator { return data_ + size_; }

  inline auto contiguous() const noexcept -> bool { return true; }

private:
  IntType size_ = 0;
  ValueType* data_ = nullptr;
};

template <typename T>
class HostArrayView2D {
public:
  using ValueType = T;
  using Iterator = T*;
  using ConstIterator = const T*;

  static constexpr IntType ORDER = 2;

  HostArrayView2D() = default;

  HostArrayView2D(ValueType* data, const IntType dimOuter, const IntType dimInner,
                  const IntType ldInner);

  HostArrayView2D(ValueType* data, const IntType dimOuter, const IntType dimInner);

  inline auto operator()(const IntType idxOuter, const IntType idxInner) -> ValueType& {
    assert(idxOuter < dims_[0]);
    assert(idxOuter >= 0);
    assert(idxInner < dims_[1]);
    assert(idxInner >= 0);
    return data_[(idxOuter * ldInner_) + idxInner];
  }

  inline auto operator()(const IntType idxOuter, const IntType idxInner) const -> const ValueType& {
    assert(idxOuter < dims_[0]);
    assert(idxOuter >= 0);
    assert(idxInner < dims_[1]);
    assert(idxInner >= 0);
    return data_[(idxOuter * ldInner_) + idxInner];
  }

  inline auto index(const IntType idxOuter, const IntType idxInner) const noexcept -> IntType {
    return (idxOuter * ldInner_) + idxInner;
  }

  inline auto data() noexcept -> ValueType* { return data_; }

  inline auto data() const noexcept -> const ValueType* { return data_; }

  inline auto empty() const noexcept -> bool { return this->size() == 0; }

  inline auto size() const noexcept -> IntType { return dims_[0] * dims_[1]; }

  inline auto dim_inner() const noexcept -> IntType { return dims_[1]; }

  inline auto dim_outer() const noexcept -> IntType { return dims_[0]; }

  inline auto ld_inner() const noexcept -> IntType { return ldInner_; }

  inline auto begin() noexcept -> Iterator { return data_; }

  inline auto begin() const noexcept -> ConstIterator { return data_; }

  inline auto cbegin() const noexcept -> ConstIterator { return data_; }

  inline auto end() noexcept -> Iterator { return data_ + size(); }

  inline auto end() const noexcept -> ConstIterator { return data_ + size(); }

  inline auto cend() const noexcept -> ConstIterator { return data_ + size(); }

  inline auto contiguous() const noexcept -> bool { return ldInner_ == dims_[1]; }

private:
  std::array<IntType, 2> dims_ = {0, 0};
  IntType ldInner_ = 0;
  ValueType* data_ = nullptr;
};

template <typename T>
class HostArrayView3D {
public:
  using ValueType = T;
  using Iterator = T*;
  using ConstIterator = const T*;

  static constexpr IntType ORDER = 3;

  HostArrayView3D() = default;

  HostArrayView3D(ValueType* data, const IntType dimOuter, const IntType dimMid,
                  const IntType dimInner, const IntType ldMid, const IntType ldInner);

  HostArrayView3D(ValueType* data, const IntType dimOuter, const IntType dimMid,
                  const IntType dimInner);

  inline auto operator()(const IntType idxOuter, const IntType idxMid,
                         const IntType idxInner) noexcept -> ValueType& {
    assert(idxOuter < dims_[0]);
    assert(idxOuter >= 0);
    assert(idxMid < dims_[1]);
    assert(idxMid >= 0);
    assert(idxInner < dims_[2]);
    assert(idxInner >= 0);
    return data_[(idxOuter * ldMid_ + idxMid) * ldInner_ + idxInner];
  }

  inline auto operator()(const IntType idxOuter, const IntType idxMid,
                         const IntType idxInner) const noexcept -> const ValueType& {
    assert(idxOuter < dims_[0]);
    assert(idxOuter >= 0);
    assert(idxMid < dims_[1]);
    assert(idxMid >= 0);
    assert(idxInner < dims_[2]);
    assert(idxInner >= 0);
    return data_[(idxOuter * ldMid_ + idxMid) * ldInner_ + idxInner];
  }

  inline auto index(const IntType idxOuter, const IntType idxMid,
                    const IntType idxInner) const noexcept -> IntType {
    return (idxOuter * ldMid_ + idxMid) * ldInner_ + idxInner;
  }

  inline auto data() noexcept -> ValueType* { return data_; }

  inline auto data() const noexcept -> const ValueType* { return data_; }

  inline auto empty() const noexcept -> bool { return this->size() == 0; }

  inline auto size() const noexcept -> IntType { return dims_[0] * dims_[1] * dims_[2]; }

  inline auto dim_inner() const noexcept -> IntType { return dims_[2]; }

  inline auto dim_mid() const noexcept -> IntType { return dims_[1]; }

  inline auto dim_outer() const noexcept -> IntType { return dims_[0]; }

  inline auto ld_inner() const noexcept -> IntType { return ldInner_; }

  inline auto ld_mid() const noexcept -> IntType { return ldMid_; }

  inline auto begin() noexcept -> Iterator { return data_; }

  inline auto begin() const noexcept -> ConstIterator { return data_; }

  inline auto cbegin() const noexcept -> ConstIterator { return data_; }

  inline auto end() noexcept -> Iterator { return data_ + size(); }

  inline auto end() const noexcept -> ConstIterator { return data_ + size(); }

  inline auto cend() const noexcept -> ConstIterator { return data_ + size(); }

  inline auto contiguous() const noexcept -> bool {
    return ldInner_ == dims_[2] && ldMid_ == dims_[1];
  }

private:
  std::array<IntType, 3> dims_ = {0, 0, 0};
  IntType ldMid_ = 0;
  IntType ldInner_ = 0;
  ValueType* data_ = nullptr;
};

// ======================
// Implementation
// ======================

template <typename T>
HostArrayView1D<T>::HostArrayView1D(ValueType* data, const IntType size)
    : size_(size), data_(data) {
  assert(!(size != 0 && data == nullptr));
}

template <typename T>
HostArrayView2D<T>::HostArrayView2D(ValueType* data, const IntType dimOuter, const IntType dimInner,
                                    const IntType ldInner)
    : dims_({dimOuter, dimInner}), ldInner_(ldInner), data_(data) {
  assert(dimInner <= ldInner);
}

template <typename T>
HostArrayView2D<T>::HostArrayView2D(ValueType* data, const IntType dimOuter, const IntType dimInner)
    : HostArrayView2D<T>(data, dimOuter, dimInner, dimInner) {}

template <typename T>
HostArrayView3D<T>::HostArrayView3D(ValueType* data, const IntType dimOuter, const IntType dimMid,
                                    const IntType dimInner, const IntType ldMid,
                                    const IntType ldInner)
    : dims_({dimOuter, dimMid, dimInner}), ldMid_(ldMid), ldInner_(ldInner), data_(data) {
  assert(dimInner <= ldInner);
  assert(dimMid <= ldMid);
}

template <typename T>
HostArrayView3D<T>::HostArrayView3D(ValueType* data, const IntType dimOuter, const IntType dimMid,
                                    const IntType dimInner)
    : HostArrayView3D<T>(data, dimOuter, dimMid, dimInner, dimMid, dimInner) {}

}  // namespace spla
#endif
