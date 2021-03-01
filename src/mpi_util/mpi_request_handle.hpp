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
#ifndef SPLA_MPI_REQUEST_HANDLE_HPP
#define SPLA_MPI_REQUEST_HANDLE_HPP

#include <mpi.h>

#include <memory>
#include <vector>

#include "mpi_util/mpi_check_status.hpp"
#include "spla/config.h"

namespace spla {

// Storage for MPI datatypes
class MPIRequestHandle {
public:
  MPIRequestHandle() = default;

  MPIRequestHandle(const MPIRequestHandle&) = delete;

  MPIRequestHandle(MPIRequestHandle&&) = default;

  auto operator=(const MPIRequestHandle& other) -> MPIRequestHandle& = delete;

  auto operator=(MPIRequestHandle&& other) -> MPIRequestHandle& = default;

  inline auto get_and_activate() -> MPI_Request* {
    activated_ = true;
    return &mpiRequest_;
  }

  inline auto is_active() -> bool { return activated_; }

  inline auto is_ready_and_active() -> bool {
    if (activated_) {
      int ready;
      mpi_check_status(MPI_Test(&mpiRequest_, &ready, MPI_STATUS_IGNORE));
      return static_cast<bool>(ready);
    }
    return false;
  }

  inline auto wait_if_active() -> void {
    if (activated_) {
      activated_ = false;
      MPI_Wait(&mpiRequest_, MPI_STATUS_IGNORE);
    }
  }

private:
  MPI_Request mpiRequest_ = MPI_REQUEST_NULL;
  bool activated_ = false;
};

}  // namespace spla

#endif
