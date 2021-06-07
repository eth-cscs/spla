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

#ifndef SPLA_SIMPLE_ALLOCATOR_HPP
#define SPLA_SIMPLE_ALLOCATOR_HPP

#include <cassert>
#include <cstddef>
#include <functional>
#include <mutex>
#include <memory>
#include <unordered_map>

#include "spla/config.h"
#include "memory/allocator.hpp"
#include "timing/timing.hpp"
#include "spla/exceptions.hpp"

namespace spla {

template <MemLoc LOCATION>
class SimpleAllocator : public Allocator<LOCATION> {
public:
  SimpleAllocator(std::function<void*(std::size_t)> allocateFunc,
                  std::function<void(void*)> deallocateFunc)
      : allocateFunc_(std::move(allocateFunc)),
        deallocateFunc_(std::move(deallocateFunc)),
        lock_(new std::mutex()),
        memorySize_(0) {
    if (!allocateFunc_ || !deallocateFunc_) {
      throw InvalidAllocatorFunctionError();
    }
  }

  SimpleAllocator(const SimpleAllocator&) = delete;

  SimpleAllocator(SimpleAllocator&&) = default;

  auto operator=(const SimpleAllocator&) -> SimpleAllocator& = delete;

  auto operator=(SimpleAllocator &&) -> SimpleAllocator& = default;

  ~SimpleAllocator() override {}

  auto allocate(std::size_t size) -> void* override {
    if (!size) return nullptr;
    std::lock_guard<std::mutex> guard(*lock_);
    SCOPED_TIMING("simple_allocate");

    void * ptr = allocateFunc_(size);
    allocatedMem_.emplace(ptr, size);
    memorySize_ += size;
    return ptr;
  }

  auto deallocate(void* ptr) -> void override {
    std::lock_guard<std::mutex> guard(*lock_);
    SCOPED_TIMING("simple_deallocate");
    deallocateFunc_(ptr);
    auto it = allocatedMem_.find(ptr);
    memorySize_ -= it->second;
    allocatedMem_.erase(it);
  }

  auto size() -> std::uint_least64_t override {
    return memorySize_;
  }

private:
  std::function<void*(std::size_t)> allocateFunc_;
  std::function<void(void*)> deallocateFunc_;

  std::unique_ptr<std::mutex> lock_;
  std::unordered_map<void*, std::size_t> allocatedMem_;
  std::uint_least64_t memorySize_;
};
}  // namespace spla

#endif
