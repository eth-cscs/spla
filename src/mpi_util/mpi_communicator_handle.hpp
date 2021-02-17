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
#ifndef SPLA_MPI_COMMUNICATOR_HANDLE_HPP
#define SPLA_MPI_COMMUNICATOR_HANDLE_HPP

#include <mpi.h>

#include <cassert>
#include <cstring>
#include <memory>

#include "mpi_util/mpi_check_status.hpp"
#include "spla/config.h"
#include "spla/exceptions.hpp"
#include "util/common_types.hpp"

namespace spla {

// Takes ownerships of a Commnunicator
class MPICommunicatorHandle {
public:
  MPICommunicatorHandle() : comm_(new MPI_Comm(MPI_COMM_SELF)), size_(1), rank_(0) {}

  explicit MPICommunicatorHandle(const MPI_Comm& comm) {
    const MPI_Comm worldComm = MPI_COMM_WORLD;
    const MPI_Comm selfComm = MPI_COMM_SELF;
    if (!std::memcmp(&comm, &worldComm, sizeof(MPI_Comm)) ||
        !std::memcmp(&comm, &selfComm, sizeof(MPI_Comm))) {
      // don't free predifned communicators
      comm_ = std::shared_ptr<MPI_Comm>(new MPI_Comm(comm));
    } else {
      comm_ = std::shared_ptr<MPI_Comm>(new MPI_Comm(comm), [](MPI_Comm* ptr) {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized) {
          MPI_Comm_free(ptr);
        }
        delete ptr;
      });
    }

    int sizeInt, rankInt;
    mpi_check_status(MPI_Comm_size(*comm_, &sizeInt));
    mpi_check_status(MPI_Comm_rank(*comm_, &rankInt));

    rank_ = static_cast<IntType>(rankInt);
    size_ = static_cast<IntType>(sizeInt);
  }

  inline auto get() const -> const MPI_Comm& { return *comm_; }

  inline auto size() const noexcept -> IntType { return size_; }

  inline auto rank() const noexcept -> IntType { return rank_; }

  inline auto clone() const -> MPICommunicatorHandle {
    MPI_Comm newComm;
    mpi_check_status(MPI_Comm_dup(this->get(), &newComm));
    return MPICommunicatorHandle(newComm);
  }

private:
  std::shared_ptr<MPI_Comm> comm_ = nullptr;
  IntType size_ = 1;
  IntType rank_ = 0;
};

}  // namespace spla

#endif
