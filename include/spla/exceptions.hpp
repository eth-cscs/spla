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
#ifndef SPLA_EXCEPTIONS_H
#define SPLA_EXCEPTIONS_H

#include <stdexcept>
#include "spla/config.h"

namespace spla {

/**
 * A generic error. Base type for all other exceptions.
 */
class SPLA_EXPORT GenericError : public std::exception {
public:
  auto what() const noexcept -> const char* override { return "SPLA: Generic error"; }
};

/**
 * Internal error.
 */
class SPLA_EXPORT InternalError : public GenericError {
public:
  auto what() const noexcept -> const char* override { return "SPLA: Internal error"; }
};

/**
 * Invalid parameter error.
 */
class SPLA_EXPORT InvalidParameterError : public GenericError {
public:
  auto what() const noexcept -> const char* override { return "SPLA: Invalid parameter error"; }
};

/**
 * Invalid pointer error.
 */
class SPLA_EXPORT InvalidPointerError : public GenericError {
public:
  auto what() const noexcept -> const char* override { return "SPLA: Invalid pointer error"; }
};

/**
 * Generic MPI Error
 */
class SPLA_EXPORT MPIError : public GenericError {
public:
  auto what() const noexcept -> const char* override { return "SPLA: MPI error"; }
};

class SPLA_EXPORT MPIAllocError : public MPIError {
public:
  auto what() const noexcept -> const char* override { return "SPLA: MPI memory allocation error"; }
};

class SPLA_EXPORT MPIThreadSupportError : public MPIError {
public:
  auto what() const noexcept -> const char* override { return "SPLA: MPI multi-threading support not sufficient"; }
};

class SPLA_EXPORT GPUError : public GenericError {
public:
  auto what() const noexcept -> const char* override { return "SPLA: GPU error"; }
};

class SPLA_EXPORT GPUSupportError : public GPUError {
public:
  auto what() const noexcept -> const char* override { return "SPLA: Not compiled with GPU support"; }
};

class SPLA_EXPORT GPUAllocationError : public GPUError {
public:
  auto what() const noexcept -> const char* override { return "SPLA: GPU allocation error"; }
};

class SPLA_EXPORT GPULaunchError : public GPUError {
public:
  auto what() const noexcept -> const char* override { return "SPLA: GPU launch error"; }
};

class SPLA_EXPORT GPUNoDeviceError : public GPUError {
public:
  auto what() const noexcept -> const char* override { return "SPLA: GPU no device error"; }
};

class SPLA_EXPORT GPUInvalidValueError : public GPUError {
public:
  auto what() const noexcept -> const char* override { return "SPLA: GPU invalid value error"; }
};

class SPLA_EXPORT GPUInvalidDevicePointerError : public GPUError {
public:
  auto what() const noexcept -> const char* override {
    return "SPLA: GPU invalid device pointer error";
  }
};

class SPLA_EXPORT GPUBlasError : public GPUError {
public:
  auto what() const noexcept -> const char* override { return "SPLA: GPU BLAS error"; }
};

}  // namespace spfft

#endif
