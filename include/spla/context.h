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
#ifndef SPLA_CONTEXT_H
#define SPLA_CONTEXT_H

#include "spla/config.h"
#include "spla/errors.h"
#include "spla/types.h"

/**
 * Context handle.
 */
typedef void* SplaContext;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Create Context with default configuration for given processing unit.
 *
 * @param[out] ctx Context handle.
 * @param[in] pu Processing unit to be used for computations.
 * @return Error code or SPLA_SUCCESS.
 */
SplaError spla_ctx_create(SplaContext* ctx, SplaProcessingUnit pu);

/**
 * Destroy context.
 *
 * @param[in] ctx Context handle.
 * @return Error code or SPLA_SUCCESS.
 */
SplaError spla_ctx_destroy(SplaContext* ctx);

/**
 * Access a Context parameter.
 * @param[in] ctx Context handle.
 * @param[out] pu Procesing unit used for computations.
 * @return Error code or SPLA_SUCCESS.
 */
SplaError spla_ctx_processing_unit(SplaContext ctx, SplaProcessingUnit* pu);

/**
 * Access a Context parameter.
 * @param[in] ctx Context handle.
 * @param[out] numThreads Maximum number of threads used for computations.
 * @return Error code or SPLA_SUCCESS.
 */
SplaError spla_ctx_num_threads(SplaContext ctx, int* numThreads);

/**
 * Access a Context parameter.
 * @param[in] ctx Context handle.
 * @param[out] numTiles Number of tiles used to overlap computation and communication.
 * @return Error code or SPLA_SUCCESS.
 */
SplaError spla_ctx_num_tiles(SplaContext ctx, int* numTiles);

/**
 * Access a Context parameter.
 * @param[in] ctx Context handle.
 * @param[out] tileSizeHost Target size of tiles on host. Used for partitioning communication.
 * @return Error code or SPLA_SUCCESS.
 */
SplaError spla_ctx_tile_size_host(SplaContext ctx, int* tileSizeHost);

/**
 * Access a Context parameter.
 * @param[in] ctx Context handle.
 * @param[out] tileSizeGPU Target size of tiles on GPU.
 * @return Error code or SPLA_SUCCESS.
 */
SplaError spla_ctx_tile_size_gpu(SplaContext ctx, int* tileSizeGPU);

/**
 * Access a Context parameter.
 * @param[in] ctx Context handle.
 * @param[out] deviceId Id of GPU used for computations. This is set as fixed parameter by query of
 * device id at context creation.
 * @return Error code or SPLA_SUCCESS.
 */
SplaError spla_ctx_gpu_device_id(SplaContext ctx, int* deviceId);

/**
 * Set the number of threads to be used.
 *
 * @param[in] ctx Context handle.
 * @param[in] numThreads Number of threads.
 * @return Error code or SPLA_SUCCESS.
 */
SplaError spla_ctx_set_num_threads(SplaContext ctx, int numThreads);

/**
 * Set the number of tiles.
 *
 * @param[in] ctx Context handle.
 * @param[in] numTiles Number of tiles.
 * @return Error code or SPLA_SUCCESS.
 */
SplaError spla_ctx_set_num_tiles(SplaContext ctx, int numTiles);

/**
 * Set the number of streams on GPU.
 *
 * @param[in] ctx Context handle.
 * @param[in] numGPUStreams Number of streams on GPU.
 * @return Error code or SPLA_SUCCESS.
 */
SplaError spla_ctx_set_num_gpu_streams(SplaContext ctx, int numGPUStreams);

/**
 * Set the target tile size used for computations.
 *
 * @param[in] ctx Context handle.
 * @param[in] tileSizeHost Target size of tiles on host. Used for partitioning communication.
 * @return Error code or SPLA_SUCCESS.
 */
SplaError spla_ctx_set_tile_size_host(SplaContext ctx, int tileSizeHost);

/**
 * Set the memory tileSizeGPU on GPU. A small mimimum is always required, therefore this is not a hard
 * tileSizeGPU.
 *
 * @param[in] ctx Context handle.
 * @param[in] tileSizeGPU Target size of tiles on GPU.
 * @return Error code or SPLA_SUCCESS.
 */
SplaError spla_ctx_set_tile_size_gpu(SplaContext ctx, int tileSizeGPU);

#ifdef __cplusplus
}
#endif
#endif
