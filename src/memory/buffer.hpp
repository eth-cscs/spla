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
#include <memory>

#include "memory/allocator.hpp"
#include "spla/config.h"
#include "spla/exceptions.hpp"

namespace spla {

template <typename T, MemLoc LOCATION>
class Buffer {
public:
  static_assert(std::is_trivially_destructible<T>::value,
                "Type in buffer must be trivially destructible! Memory is not initialized and "
                "destructors are not "
                "called!");

  Buffer(std::shared_ptr<Allocator<LOCATION>> allocator, std::size_t size)
      : size_(0), data_(nullptr), allocator_(std::move(allocator)) {
    assert(allocator_);
    this->resize(size);
  }

  Buffer(const Buffer&) = delete;

  Buffer(Buffer&& buffer) { *this = std::move(buffer); }

  ~Buffer() {
    if (allocator_ && size_) {
      allocator_->deallocate(data_);
    }
  }

  auto operator=(const Buffer&) -> Buffer& = delete;

  auto operator=(Buffer&& buffer) -> Buffer& {
    this->resize(0);
    size_ = buffer.size_;
    data_ = buffer.data_;
    allocator_ = std::move(buffer.allocator_);
    buffer.size_ = 0;
    buffer.data_ = nullptr;
    return *this;
  }

  auto empty() const noexcept -> bool { return !data_; }

  auto size() const noexcept -> std::size_t { return size_; }

  auto data() -> T* {
    assert(data_);
    return reinterpret_cast<T*>(data_);
  }

  auto data() const -> const T* {
    assert(data_);
    return reinterpret_cast<T*>(data_);
  }

  auto resize(std::size_t newSize) -> void {
    if (newSize != size_) {
      if (newSize == 0) {
        if (size_) allocator_->deallocate(data_);
        size_ = 0;
        data_ = nullptr;
      } else {
        if (size_) allocator_->deallocate(data_);
        data_ = allocator_->allocate(newSize * sizeof(T));
        size_ = newSize;
      }
    }
  }

protected:
  std::size_t size_ = 0;
  void* data_ = nullptr;
  std::shared_ptr<Allocator<LOCATION>> allocator_;
};
}  // namespace spla

#endif
