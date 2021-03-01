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

#ifndef SPLA_GPU_TRANSFER_HPP
#define SPLA_GPU_TRANSFER_HPP

#include <cassert>
#include <type_traits>

#include "gpu_util/gpu_runtime_api.hpp"
#include "gpu_util/gpu_stream_handle.hpp"
#include "memory/gpu_array_const_view.hpp"
#include "memory/gpu_array_view.hpp"
#include "memory/host_array_const_view.hpp"
#include "memory/host_array_view.hpp"
#include "spla/config.h"
#include "util/common_types.hpp"

namespace spla {

template <typename T, typename U>
auto copy_to_gpu(const HostArrayConstView1D<T>& source, GPUArrayView1D<U>& target) -> void {
  static_assert(sizeof(T) == sizeof(U), "Size of types for GPU transfer must match!");
  assert(source.size() == static_cast<IntType>(target.size()));
  gpu::check_status(gpu::memcpy(static_cast<void*>(target.data()),
                                static_cast<const void*>(source.data()), source.size() * sizeof(T),
                                gpu::flag::MemcpyHostToDevice));
}

template <typename T, typename U>
auto copy_to_gpu(const HostArrayConstView1D<T>& source, GPUArrayView1D<U>&& target) -> void {
  copy_to_gpu(source, target);
}

template <typename T, typename U>
auto copy_to_gpu_async(const gpu::StreamType& stream, const HostArrayConstView1D<T>& source,
                       GPUArrayView1D<U>& target) -> void {
  static_assert(sizeof(T) == sizeof(U), "Size of types for GPU transfer must match!");
  assert(source.size() == static_cast<IntType>(target.size()));
  gpu::check_status(
      gpu::memcpy_async(static_cast<void*>(target.data()), static_cast<const void*>(source.data()),
                        source.size() * sizeof(T), gpu::flag::MemcpyHostToDevice, stream));
}

template <typename T, typename U>
auto copy_to_gpu_async(const gpu::StreamType& stream, const HostArrayConstView1D<T>& source,
                       GPUArrayView1D<U>&& target) -> void {
  copy_to_gpu_async(stream, source, target);
}

template <typename T, typename U>
auto copy_to_gpu(const HostArrayConstView2D<T>& source, GPUArrayView2D<U>& target) -> void {
  static_assert(sizeof(T) == sizeof(U), "Size of types for GPU transfer must match!");
  assert(source.dim_inner() == static_cast<IntType>(target.dim_inner()));
  assert(source.dim_outer() == static_cast<IntType>(target.dim_outer()));

  if (source.contiguous() && target.contiguous()) {
    gpu::check_status(gpu::memcpy(static_cast<void*>(target.data()),
                                  static_cast<const void*>(source.data()),
                                  source.size() * sizeof(T), gpu::flag::MemcpyHostToDevice));
  } else {
    gpu::check_status(gpu::memcpy_2d(
        static_cast<void*>(target.data()), target.ld_inner() * sizeof(U),
        static_cast<const void*>(source.data()), source.ld_inner() * sizeof(T),
        source.dim_inner() * sizeof(T), source.dim_outer(), gpu::flag::MemcpyHostToDevice));
  }
}

template <typename T, typename U>
auto copy_to_gpu(const HostArrayConstView2D<T>& source, GPUArrayView2D<U>&& target) -> void {
  copy_to_gpu(source, target);
}

template <typename T, typename U>
auto copy_to_gpu_async(const gpu::StreamType& stream, const HostArrayConstView2D<T>& source,
                       GPUArrayView2D<U>& target) -> void {
  static_assert(sizeof(T) == sizeof(U), "Size of types for GPU transfer must match!");
  assert(source.dim_inner() == static_cast<IntType>(target.dim_inner()));
  assert(source.dim_outer() == static_cast<IntType>(target.dim_outer()));

  if (source.contiguous() && target.contiguous()) {
    gpu::check_status(gpu::memcpy_async(
        static_cast<void*>(target.data()), static_cast<const void*>(source.data()),
        source.size() * sizeof(T), gpu::flag::MemcpyHostToDevice, stream));
  } else {
    gpu::check_status(gpu::memcpy_2d_async(
        static_cast<void*>(target.data()), target.ld_inner() * sizeof(U),
        static_cast<const void*>(source.data()), source.ld_inner() * sizeof(T),
        source.dim_inner() * sizeof(T), source.dim_outer(), gpu::flag::MemcpyHostToDevice, stream));
  }
}

template <typename T, typename U>
auto copy_to_gpu_async(const gpu::StreamType& stream, const HostArrayConstView2D<T>& source,
                       GPUArrayView2D<U>&& target) -> void {
  copy_to_gpu_async(stream, source, target);
}

template <typename T, typename U>
auto copy_from_gpu(const GPUArrayConstView1D<T>& source, HostArrayView1D<U>& target) -> void {
  static_assert(sizeof(T) == sizeof(U), "Size of types for GPU transfer must match!");
  assert(source.size() == static_cast<IntType>(target.size()));
  gpu::check_status(gpu::memcpy(static_cast<void*>(target.data()),
                                static_cast<const void*>(source.data()), source.size() * sizeof(T),
                                gpu::flag::MemcpyDeviceToHost));
}

template <typename T, typename U>
auto copy_from_gpu(const GPUArrayConstView1D<T>& source, HostArrayView1D<U>&& target) -> void {
  copy_from_gpu(source, target);
}

template <typename T, typename U>
auto copy_from_gpu_async(const gpu::StreamType& stream, const GPUArrayConstView1D<T>& source,
                         HostArrayView1D<U>& target) -> void {
  static_assert(sizeof(T) == sizeof(U), "Size of types for GPU transfer must match!");
  assert(source.size() == static_cast<IntType>(target.size()));
  gpu::check_status(
      gpu::memcpy_async(static_cast<void*>(target.data()), static_cast<const void*>(source.data()),
                        source.size() * sizeof(T), gpu::flag::MemcpyDeviceToHost, stream));
}

template <typename T, typename U>
auto copy_from_gpu_async(const gpu::StreamType& stream, const GPUArrayConstView1D<T>& source,
                         HostArrayView1D<U>&& target) -> void {
  copy_from_gpu_async(stream, source, target);
}

template <typename T, typename U>
auto copy_from_gpu(const GPUArrayConstView2D<T>& source, HostArrayView2D<U>& target) -> void {
  static_assert(sizeof(T) == sizeof(U), "Size of types for GPU transfer must match!");
  assert(source.dim_inner() == static_cast<IntType>(target.dim_inner()));
  assert(source.dim_outer() == static_cast<IntType>(target.dim_outer()));

  if (source.contiguous() && target.contiguous()) {
    gpu::check_status(gpu::memcpy(static_cast<void*>(target.data()),
                                  static_cast<const void*>(source.data()),
                                  source.size() * sizeof(T), gpu::flag::MemcpyDeviceToHost));
  } else {
    gpu::check_status(gpu::memcpy_2d(
        static_cast<void*>(target.data()), target.ld_inner() * sizeof(U),
        static_cast<const void*>(source.data()), source.ld_inner() * sizeof(T),
        source.dim_inner() * sizeof(T), source.dim_outer(), gpu::flag::MemcpyDeviceToHost));
  }
}

template <typename T, typename U>
auto copy_from_gpu(const GPUArrayConstView2D<T>& source, HostArrayView2D<U>&& target) -> void {
  copy_from_gpu(source, target);
}

template <typename T, typename U>
auto copy_from_gpu_async(const gpu::StreamType& stream, const GPUArrayConstView2D<T>& source,
                         HostArrayView2D<U>& target) -> void {
  static_assert(sizeof(T) == sizeof(U), "Size of types for GPU transfer must match!");
  assert(source.dim_inner() == static_cast<IntType>(target.dim_inner()));
  assert(source.dim_outer() == static_cast<IntType>(target.dim_outer()));

  if (source.contiguous() && target.contiguous()) {
    gpu::check_status(gpu::memcpy_async(
        static_cast<void*>(target.data()), static_cast<const void*>(source.data()),
        source.size() * sizeof(T), gpu::flag::MemcpyDeviceToHost, stream));
  } else {
    gpu::check_status(gpu::memcpy_2d_async(
        static_cast<void*>(target.data()), target.ld_inner() * sizeof(U),
        static_cast<const void*>(source.data()), source.ld_inner() * sizeof(T),
        source.dim_inner() * sizeof(T), source.dim_outer(), gpu::flag::MemcpyDeviceToHost, stream));
  }
}

template <typename T, typename U>
auto copy_from_gpu_async(const gpu::StreamType& stream, const GPUArrayConstView2D<T>& source,
                         HostArrayView2D<U>&& target) -> void {
  copy_from_gpu_async(stream, source, target);
}

}  // namespace spla

#endif
