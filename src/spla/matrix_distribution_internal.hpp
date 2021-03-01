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
#ifndef SPLA_MATRIX_DISTRIBUTION_INTERNAL_HPP
#define SPLA_MATRIX_DISTRIBUTION_INTERNAL_HPP

#include <deque>

#include "mpi_util/mpi_communicator_handle.hpp"
#include "spla/context.hpp"
#include "spla/exceptions.hpp"
#include "spla/matrix_distribution.hpp"
#include "util/common_types.hpp"

namespace spla {

class MatrixDistributionInternal {
public:
  inline auto proc_grid_rows() const -> IntType { return procGridRows_; }

  inline auto proc_grid_cols() const -> IntType { return procGridCols_; }

  inline auto row_block_size() const -> IntType { return rowBlockSize_; }

  inline auto col_block_size() const -> IntType { return colBlockSize_; }

  inline auto comm() const -> const MPICommunicatorHandle& { return comms_.front(); }

  inline auto get_comms(IntType numComms) -> const std::deque<MPICommunicatorHandle>& {
    if (comms_.size() < numComms) {
      for (IntType i = static_cast<IntType>(comms_.size()); i < numComms; ++i) {
        comms_.emplace_back(comms_.front().clone());
      }
    }
    return comms_;
  }

  inline auto type() const -> SplaDistributionType { return type_; }

  static auto create_blacs_block_cyclic(MPI_Comm comm, char order, IntType procGridRows,
                                        IntType procGridCols, IntType rowBlockSize,
                                        IntType colBlockSize) -> MatrixDistributionInternal;

  static auto create_blacs_block_cyclic_from_mapping(MPI_Comm comm, const int* mapping,
                                                     IntType procGridRows, IntType procGridCols,
                                                     IntType rowBlockSize, IntType colBlockSize)
      -> MatrixDistributionInternal;

  static auto create_mirror(MPI_Comm comm) -> MatrixDistributionInternal;

private:
  MatrixDistributionInternal(MPI_Comm comm, const int* mapping, IntType procGridRows,
                             IntType procGridCols, IntType rowBlockSize, IntType colBlockSize);

  explicit MatrixDistributionInternal(MPI_Comm comm);

  SplaDistributionType type_;
  std::deque<MPICommunicatorHandle> comms_;  // use deque to avoid dangling references upon resizing
  IntType procGridRows_, procGridCols_;
  IntType rowBlockSize_, colBlockSize_;
};

}  // namespace spla

#endif
