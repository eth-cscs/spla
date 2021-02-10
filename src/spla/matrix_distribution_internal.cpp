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

#include "spla/matrix_distribution_internal.hpp"

#include <cstring>
#include <functional>
#include <set>
#include <vector>

#include "memory/host_array_view.hpp"
#include "mpi_util/mpi_communicator_handle.hpp"
#include "spla/exceptions.hpp"
#include "spla/matrix_distribution.hpp"

namespace spla {

auto MatrixDistributionInternal::create_blacs_block_cyclic(
    MPI_Comm comm, char order, IntType procGridRows, IntType procGridCols, IntType rowBlockSize,
    IntType colBlockSize) -> MatrixDistributionInternal {
  if (order != 'R' && order != 'r' && order != 'C' && order != 'c') throw InvalidParameterError();
  if (procGridRows < 1 || procGridCols < 1 || rowBlockSize < 1 || colBlockSize < 1)
    throw InvalidParameterError();

  std::vector<int> mapping(procGridCols * procGridRows);
  HostArrayView2D<int> mappingView(mapping.data(), procGridCols, procGridRows);

  // reorder ranks, such that they can be adressed by row-major convention
  int counter = 0;
  if (order == 'R' || order == 'r') {
    for (IntType r = 0; r < procGridRows; ++r) {
      for (IntType c = 0; c < procGridCols; ++c, ++counter) {
        mappingView(c, r) = counter;
      }
    }
  } else {
    for (IntType c = 0; c < procGridCols; ++c) {
      for (IntType r = 0; r < procGridRows; ++r, ++counter) {
        mappingView(c, r) = counter;
      }
    }
  }

  return MatrixDistributionInternal(comm, mapping.data(), procGridRows, procGridCols, rowBlockSize,
                                    colBlockSize);
}

auto MatrixDistributionInternal::create_blacs_block_cyclic_from_mapping(
    MPI_Comm comm, const int *mapping, IntType procGridRows, IntType procGridCols,
    IntType rowBlockSize, IntType colBlockSize) -> MatrixDistributionInternal {
  if (procGridRows < 1 || procGridCols < 1 || rowBlockSize < 1 || colBlockSize < 1)
    throw InvalidParameterError();
  if (!mapping) throw InvalidParameterError();
  return MatrixDistributionInternal(comm, mapping, procGridRows, procGridCols, rowBlockSize,
                                    colBlockSize);
}

auto MatrixDistributionInternal::create_mirror(MPI_Comm comm) -> MatrixDistributionInternal {
  return MatrixDistributionInternal(comm);
}

MatrixDistributionInternal::MatrixDistributionInternal(MPI_Comm comm, const int *mapping,
                                                       IntType procGridRows, IntType procGridCols,
                                                       IntType rowBlockSize, IntType colBlockSize)
    : type_(SplaDistributionType::SPLA_DIST_BLACS_BLOCK_CYCLIC),
      procGridRows_(procGridRows),
      procGridCols_(procGridCols),
      rowBlockSize_(rowBlockSize),
      colBlockSize_(colBlockSize) {
  if (procGridRows < 1 || procGridCols < 1 || rowBlockSize < 1 || colBlockSize < 1)
    throw InvalidParameterError();
  if (!mapping) throw InvalidParameterError();

  int commSizeInt = 0;
  mpi_check_status(MPI_Comm_size(comm, &commSizeInt));
  IntType commSize = static_cast<IntType>(commSizeInt);
  std::vector<int> fullMapping(commSize);
  std::set<IntType> mappedRanks;
  for (IntType i = 0; i < procGridRows_ * procGridCols_; ++i) {
    if (mapping[i] < 0 || mapping[i] >= commSize) {
      throw std::runtime_error("Invalid rank mapping");
    }
    if (!mappedRanks.insert(mapping[i]).second) {
      throw std::runtime_error("Rank mapping duplication");
    }

    fullMapping[i] = mapping[i];
  }

  // assigned unused labels to excess ranks
  IntType excessRankIdx = procGridRows_ * procGridCols_;
  for (IntType i = 0; i < commSize; ++i) {
    if (!mappedRanks.count(i)) {
      fullMapping[excessRankIdx] = i;
      ++excessRankIdx;
    }
  }
  if (excessRankIdx != commSize) {
    throw std::runtime_error("Rank mapping failed");
  }

  auto groupDeleteFunc = [](MPI_Group *group) -> void {
    MPI_Group_free(group);
    delete group;
  };

  std::unique_ptr<MPI_Group, std::function<void(MPI_Group *)>> originGroup(
      new MPI_Group(MPI_GROUP_EMPTY), groupDeleteFunc);
  mpi_check_status(MPI_Comm_group(comm, originGroup.get()));

  std::unique_ptr<MPI_Group, std::function<void(MPI_Group *)>> newGroup(
      new MPI_Group(MPI_GROUP_EMPTY), groupDeleteFunc);
  mpi_check_status(
      MPI_Group_incl(*originGroup, fullMapping.size(), fullMapping.data(), newGroup.get()));

  // create communicator with new rank order
  MPI_Comm newComm = MPI_COMM_NULL;
  mpi_check_status(MPI_Comm_create_group(comm, *newGroup, 0, &newComm));

  if (newComm == MPI_COMM_NULL) {
    throw MPIError();
  }

  comms_.emplace_back(newComm);
}

MatrixDistributionInternal::MatrixDistributionInternal(MPI_Comm comm)
    : type_(SplaDistributionType::SPLA_DIST_MIRROR),
      procGridRows_(1),
      procGridCols_(1),
      rowBlockSize_(256),
      colBlockSize_(256) {
  const MPI_Comm selfComm = MPI_COMM_SELF;
  if (!std::memcmp(&comm, &selfComm, sizeof(MPI_Comm))) {
    // don't duplicate self communicator
    comms_.emplace_back(comm);
  } else {
    MPI_Comm newComm;
    mpi_check_status(MPI_Comm_dup(comm, &newComm));
    comms_.emplace_back(newComm);
  }
  procGridRows_ = comms_.front().size();
}

}  // namespace spla
