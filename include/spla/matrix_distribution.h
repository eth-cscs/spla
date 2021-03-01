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
#ifndef SPLA_MATRIX_DISTRIBUTION_H
#define SPLA_MATRIX_DISTRIBUTION_H

#include <mpi.h>

#include "spla/config.h"
#include "spla/errors.h"
#include "spla/types.h"

/**
 * Matrix distribution handle.
 */
typedef void* SplaMatrixDistribution;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Create a blacs block cyclic matrix distribution with row major or coloumn major ordering of MPI
 * ranks.
 *
 * @param[out] matDis Matrix distribution handle.
 * @param[in] comm MPI communicator to be used.
 * @param[in] order Either 'R' for row major ordering or 'C' for coloumn major ordering.
 * @param[in] procGridRows Number of rows in process grid.
 * @param[in] procGridCols Number of coloumns in process grid.
 * @param[in] rowBlockSize Row block size for matrix partitioning.
 * @param[in] colBlockSize Coloumn block size for matrix partitioning.
 * @return Error code or SPLA_SUCCESS.
 */
SplaError spla_mat_dis_create_block_cyclic(SplaMatrixDistribution* matDis, MPI_Comm comm,
                                           char order, int procGridRows, int procGridCols,
                                           int rowBlockSize, int colBlockSize);

/**
 * Create a blacs block cyclic matrix distribution with given process grid mapping.
 *
 * @param[out] matDis Matrix distribution handle.
 * @param[in] comm MPI communicator to be used.
 * @param[in] mapping Pointer to array of size procGridRows * procGridCols mapping MPI ranks onto
 * a coloumn major process grid.
 * @param[in] procGridRows Number of rows in process grid.
 * @param[in] procGridCols Number of coloumns in process grid.
 * @param[in] rowBlockSize Row block size for matrix partitioning.
 * @param[in] colBlockSize Coloumn block size for matrix partitioning.
 * @return Error code or SPLA_SUCCESS.
 */
SplaError spla_mat_dis_create_blacs_block_cyclic_from_mapping(SplaMatrixDistribution* matDis,
                                                              MPI_Comm comm, const int* mapping,
                                                              int procGridRows, int procGridCols,
                                                              int rowBlockSize, int colBlockSize);

/**
 * Destroy matrix distribution.
 *
 * @param[in] matDis Matrix distribution handle.
 * @return Error code or SPLA_SUCCESS.
 */
SplaError spla_mat_dis_destroy(SplaMatrixDistribution* matDis);

/**
 * Create a mirror distribution, where the full matrix is stored on each MPI rank.
 *
 * @param[out] matDis Matrix distribution handle.
 * @param[in] comm MPI communicator to be used.
 * @return Error code or SPLA_SUCCESS.
 */
SplaError spla_mat_dis_create_mirror(SplaMatrixDistribution* matDis, MPI_Comm comm);

/**
 * Access a distribution parameter.
 *
 * @param[in] matDis Matrix distribution handle.
 * @param[out] procGridRows Number of rows in process grid.
 * @return Error code or SPLA_SUCCESS.
 */
SplaError spla_mat_dis_proc_grid_rows(SplaMatrixDistribution matDis, int* procGridRows);

/**
 * Access a distribution parameter.
 *
 * @param[in] matDis Matrix distribution handle.
 * @param[out] procGridCols Number of coloumns in process grid.
 * @return Error code or SPLA_SUCCESS.
 */
SplaError spla_mat_dis_proc_grid_cols(SplaMatrixDistribution matDis, int* procGridCols);

/**
 * Access a distribution parameter.
 *
 * @param[in] matDis Matrix distribution handle.
 * @param[out] rowBlockSize Row block size used for matrix partitioning.
 * @return Error code or SPLA_SUCCESS.
 */
SplaError spla_mat_dis_row_block_size(SplaMatrixDistribution matDis, int* rowBlockSize);

/**
 * Access a distribution parameter.
 *
 * @param[in] matDis Matrix distribution handle.
 * @param[out] colBlockSize Coloumn block size used for matrix partitioning.
 * @return Error code or SPLA_SUCCESS.
 */
SplaError spla_mat_dis_col_block_size(SplaMatrixDistribution matDis, int* colBlockSize);

/**
 * Access a distribution parameter.
 *
 * @param[in] matDis Matrix distribution handle.
 * @param[out] type Distribution type
 * @return Error code or SPLA_SUCCESS.
 */
SplaError spla_mat_dis_type(SplaMatrixDistribution matDis, SplaDistributionType* type);

/**
 * Access a distribution parameter.
 *
 * @param[in] matDis Matrix distribution handle.
 * @param[out] comm Communicator used internally. Order of ranks may differ from communicator
 * provided for creation of distribution.
 * @return Error code or SPLA_SUCCESS.
 */
SplaError spla_mat_dis_comm(SplaMatrixDistribution matDis, MPI_Comm* comm);
#ifdef __cplusplus
}
#endif
#endif
