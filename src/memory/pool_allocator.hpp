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

#ifndef SPLA_POOL_ALLOCATOR_HPP
#define SPLA_POOL_ALLOCATOR_HPP

#include <cassert>
#include <cstddef>
#include <map>
#include <memory>
#include <unordered_map>
#include <functional>
#include <mutex>

#include "spla/config.h"
#include "memory/allocator.hpp"
#include "timing/timing.hpp"
#include "spla/exceptions.hpp"

namespace spla {

template <MemLoc LOCATION>
class PoolAllocator : public Allocator<LOCATION> {
public:
  PoolAllocator(std::function<void*(std::size_t)> allocateFunc,
                std::function<void(void*)> deallocateFunc)
      : allocateFunc_(std::move(allocateFunc)),
        deallocateFunc_(std::move(deallocateFunc)),
        lock_(new std::mutex()) {
    if (!allocateFunc_ || !deallocateFunc_) {
      throw InvalidAllocatorFunctionError();
    }
  }

  PoolAllocator(const PoolAllocator&) = delete;

  PoolAllocator(PoolAllocator&&) = default;

  auto operator=(const PoolAllocator&) -> PoolAllocator& = delete;

  auto operator=(PoolAllocator&&) -> PoolAllocator& = default;

  ~PoolAllocator() override {
    for (auto& pair : allocatedMem_) {
      assert(false);  // No allocated memory should still exist with correct usage
      deallocateFunc_(pair.first);
    }
    for (auto& pair : freeMem_) {
      deallocateFunc_(pair.second);
    }
  }

  auto allocate(std::size_t size) -> void* override {
    if (!size) return nullptr;
    std::lock_guard<std::mutex> guard(*lock_);
    SCOPED_TIMING("pool_allocate");

    void* ptr = nullptr;
    // find block which is greater or equal to size
    auto boundIt = freeMem_.lower_bound(size);

    if (boundIt == freeMem_.end()) {
      // No memory block is large enough. Free the largest one and allocate new size.
      if(!freeMem_.empty()) {
        auto backIt = --freeMem_.end();
        deallocateFunc_(backIt->second);
        freeMem_.erase(backIt);
      }
      ptr = allocateFunc_(size);
      allocatedMem_.emplace(ptr, size);
    } else {
      // Use already allocated memory block.
      ptr = boundIt->second;
      allocatedMem_.emplace(boundIt->second, boundIt->first);
      freeMem_.erase(boundIt);
    }

    return ptr;
  }

  auto deallocate(void* ptr) -> void override {
    std::lock_guard<std::mutex> guard(*lock_);
    SCOPED_TIMING("pool_deallocate");

    auto it = allocatedMem_.find(ptr);
    assert(it != allocatedMem_.end());  // avoid throwing exception when deallocating
    if (it != allocatedMem_.end()) {
      freeMem_.emplace(it->second, it->first);
      allocatedMem_.erase(it);
    }
  }

private:
  std::function<void*(std::size_t)> allocateFunc_;
  std::function<void(void*)> deallocateFunc_;

  std::multimap<std::size_t, void*> freeMem_;
  std::unordered_map<void*, std::size_t> allocatedMem_;

  std::unique_ptr<std::mutex> lock_;
};
}  // namespace spla

#endif
