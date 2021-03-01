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

#ifndef SPLA_BUFFER_HPP
#define SPLA_BUFFER_HPP

#include <cassert>
#include <cstddef>

#include "spla/config.h"
#include "spla/exceptions.hpp"

namespace spla {

// Strictly growing memory buffer for trivial types
template <class ALLOCATOR>
class Buffer {
public:
  Buffer() noexcept : data_(nullptr), sizeInBytes_(0) {}

  Buffer(const Buffer&) = delete;

  Buffer(Buffer&& buffer) noexcept : Buffer() { *this = std::move(buffer); }

  ~Buffer() { this->deallocate(); }

  auto operator=(const Buffer&) -> Buffer& = delete;

  auto operator=(Buffer&& buffer) noexcept -> Buffer& {
    this->deallocate();
    this->data_ = buffer.data_;
    this->sizeInBytes_ = buffer.sizeInBytes_;
    buffer.data_ = nullptr;
    buffer.sizeInBytes_ = 0;
    return *this;
  }

  auto empty() const -> bool { return !data_; }

  template <typename T>
  auto resize(std::size_t size) -> void {
    static_assert(std::is_trivially_destructible<T>::value,
                  "Type in buffer must be trivially destructible! Memory is not initialized and "
                  "destructors are not "
                  "called!");
    if (sizeInBytes_ < size * sizeof(T)) {
      this->deallocate();
      const auto targetSize = size * sizeof(T);
      data_ = ALLOCATOR::allocate(targetSize);
      assert(data_);
      sizeInBytes_ = targetSize;
    }
  }

  template <typename T>
  auto size() -> std::size_t {
    return sizeInBytes_ / sizeof(T);
  }

  template <typename T>
  auto data() -> T* {
    static_assert(std::is_trivially_destructible<T>::value,
                  "Type in buffer must be trivially destructible! Memory is not initialized and "
                  "destructors are not "
                  "called!");
    assert(data_);
    return reinterpret_cast<T*>(data_);
  }

  auto deallocate() noexcept(noexcept(ALLOCATOR::deallocate(nullptr))) -> void {
    if (data_) {
      ALLOCATOR::deallocate(this->data_);
    }
    sizeInBytes_ = 0;
    data_ = nullptr;
  }

private:
  std::size_t sizeInBytes_ = 0;
  void* data_ = nullptr;
};

}  // namespace spla

#endif
