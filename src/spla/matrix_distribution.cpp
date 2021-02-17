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

#include "spla/matrix_distribution.hpp"

#include <memory>

#include "mpi_util/mpi_communicator_handle.hpp"
#include "spla/matrix_distribution.h"
#include "spla/matrix_distribution_internal.hpp"

namespace spla {

MatrixDistribution MatrixDistribution::create_blacs_block_cyclic(MPI_Comm comm, char order,
                                                                 int procGridRows, int procGridCols,
                                                                 int rowBlockSize,
                                                                 int colBlockSize) {
  return MatrixDistribution(std::make_shared<MatrixDistributionInternal>(
      MatrixDistributionInternal::create_blacs_block_cyclic(comm, order, procGridRows, procGridCols,
                                                            rowBlockSize, colBlockSize)));
}

MatrixDistribution MatrixDistribution::create_blacs_block_cyclic_from_mapping(
    MPI_Comm comm, const int* mapping, int procGridRows, int procGridCols, int rowBlockSize,
    int colBlockSize) {
  return MatrixDistribution(std::make_shared<MatrixDistributionInternal>(
      MatrixDistributionInternal::create_blacs_block_cyclic_from_mapping(
          comm, mapping, procGridRows, procGridCols, rowBlockSize, colBlockSize)));
}

MatrixDistribution MatrixDistribution::create_mirror(MPI_Comm comm) {
  return MatrixDistribution(std::make_shared<MatrixDistributionInternal>(
      MatrixDistributionInternal::create_mirror(comm)));
}

MatrixDistribution::MatrixDistribution(std::shared_ptr<MatrixDistributionInternal> descInternal)
    : descInternal_(std::move(descInternal)) {}

int MatrixDistribution::proc_grid_rows() const { return descInternal_->proc_grid_rows(); }

int MatrixDistribution::proc_grid_cols() const { return descInternal_->proc_grid_cols(); }

int MatrixDistribution::row_block_size() const { return descInternal_->row_block_size(); }

int MatrixDistribution::col_block_size() const { return descInternal_->col_block_size(); }

SplaDistributionType MatrixDistribution::type() const { return descInternal_->type(); }

MPI_Comm MatrixDistribution::comm() { return descInternal_->comm().get(); }

}  // namespace spla

extern "C" {

SplaError spla_mat_dis_create_block_cyclic(SplaMatrixDistribution* matDis, MPI_Comm comm,
                                           char order, int procGridRows, int procGridCols,
                                           int rowBlockSize, int colBlockSize) {
  try {
    *matDis = reinterpret_cast<void*>(
        new spla::MatrixDistribution(spla::MatrixDistribution::create_blacs_block_cyclic(
            comm, order, procGridRows, procGridCols, rowBlockSize, colBlockSize)));
  } catch (const spla::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SplaError::SPLA_UNKNOWN_ERROR;
  }
  return SplaError::SPLA_SUCCESS;
}

SplaError spla_mat_dis_create_blacs_block_cyclic_from_mapping(SplaMatrixDistribution* matDis,
                                                              MPI_Comm comm, const int* mapping,
                                                              int procGridRows, int procGridCols,
                                                              int rowBlockSize, int colBlockSize) {
  try {
    *matDis = reinterpret_cast<void*>(new spla::MatrixDistribution(
        spla::MatrixDistribution::create_blacs_block_cyclic_from_mapping(
            comm, mapping, procGridRows, procGridCols, rowBlockSize, colBlockSize)));
  } catch (const spla::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SplaError::SPLA_UNKNOWN_ERROR;
  }
  return SplaError::SPLA_SUCCESS;
}

SplaError spla_mat_dis_destroy(SplaMatrixDistribution* matDis) {
  if (!matDis) {
    return SplaError::SPLA_INVALID_HANDLE_ERROR;
  }
  try {
    delete reinterpret_cast<spla::MatrixDistribution*>(*matDis);
  } catch (const spla::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SplaError::SPLA_UNKNOWN_ERROR;
  }
  *matDis = nullptr;
  return SplaError::SPLA_SUCCESS;
}

SplaError spla_create_mirror(SplaMatrixDistribution* matDis, MPI_Comm comm) {
  try {
    *matDis = reinterpret_cast<void*>(
        new spla::MatrixDistribution(spla::MatrixDistribution::create_mirror(comm)));
  } catch (const spla::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SplaError::SPLA_UNKNOWN_ERROR;
  }
  return SplaError::SPLA_SUCCESS;
}

SplaError spla_mat_dis_proc_grid_rows(SplaMatrixDistribution matDis, int* procGridRows) {
  try {
    *procGridRows = reinterpret_cast<spla::MatrixDistribution*>(matDis)->proc_grid_rows();
  } catch (const spla::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SplaError::SPLA_UNKNOWN_ERROR;
  }
  return SplaError::SPLA_SUCCESS;
}

SplaError spla_mat_dis_proc_grid_cols(SplaMatrixDistribution matDis, int* procGridCols) {
  try {
    *procGridCols = reinterpret_cast<spla::MatrixDistribution*>(matDis)->proc_grid_cols();
  } catch (const spla::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SplaError::SPLA_UNKNOWN_ERROR;
  }
  return SplaError::SPLA_SUCCESS;
}

SplaError spla_mat_dis_row_block_size(SplaMatrixDistribution matDis, int* rowBlockSize) {
  try {
    *rowBlockSize = reinterpret_cast<spla::MatrixDistribution*>(matDis)->row_block_size();
  } catch (const spla::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SplaError::SPLA_UNKNOWN_ERROR;
  }
  return SplaError::SPLA_SUCCESS;
}

SplaError spla_mat_dis_col_block_size(SplaMatrixDistribution matDis, int* colBlockSize) {
  try {
    *colBlockSize = reinterpret_cast<spla::MatrixDistribution*>(matDis)->col_block_size();
  } catch (const spla::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SplaError::SPLA_UNKNOWN_ERROR;
  }
  return SplaError::SPLA_SUCCESS;
}

SplaError spla_mat_dis_type(SplaMatrixDistribution matDis, SplaDistributionType* type) {
  try {
    *type = reinterpret_cast<spla::MatrixDistribution*>(matDis)->type();
  } catch (const spla::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SplaError::SPLA_UNKNOWN_ERROR;
  }
  return SplaError::SPLA_SUCCESS;
}

SplaError spla_mat_dis_comm(SplaMatrixDistribution matDis, MPI_Comm* comm) {
  try {
    *comm = reinterpret_cast<spla::MatrixDistribution*>(matDis)->comm();
  } catch (const spla::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SplaError::SPLA_UNKNOWN_ERROR;
  }
  return SplaError::SPLA_SUCCESS;
}
}
