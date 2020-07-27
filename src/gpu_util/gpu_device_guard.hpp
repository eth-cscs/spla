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
#ifndef SPLA_GPU_DEVICE_GUARD_HPP
#define SPLA_GPU_DEVICE_GUARD_HPP

#include "spla/config.h"
#include <memory>
#include "gpu_util/gpu_runtime_api.hpp"
#include "spla/exceptions.hpp"
namespace spla {
class GPUDeviceGuard {
public:
  explicit GPUDeviceGuard(const int deviceId) : targetDeviceId_(deviceId), originalDeviceId_(0) {
    gpu::check_status(gpu::get_device(&originalDeviceId_));
    if (originalDeviceId_ != deviceId) {
      gpu::check_status(gpu::set_device(deviceId));
    }
  };

  GPUDeviceGuard() = delete;
  GPUDeviceGuard(const GPUDeviceGuard&) = delete;
  GPUDeviceGuard(GPUDeviceGuard&&) = delete;
  auto operator=(const GPUDeviceGuard&) -> GPUDeviceGuard& = delete;
  auto operator=(GPUDeviceGuard &&) -> GPUDeviceGuard& = delete;

  ~GPUDeviceGuard() {
    if (targetDeviceId_ != originalDeviceId_) {
      gpu::set_device(originalDeviceId_);  // no check to avoid throw exeception in destructor
    }
  }

private:
  int targetDeviceId_ = 0;
  int originalDeviceId_ = 0;
};
}  // namespace spla

#endif
