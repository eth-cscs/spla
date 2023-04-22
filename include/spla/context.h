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

#include <stddef.h>
#include <stdint.h>

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
SPLA_EXPORT SplaError spla_ctx_create(SplaContext* ctx, SplaProcessingUnit pu);

/**
 * Destroy context.
 *
 * @param[in] ctx Context handle.
 * @return Error code or SPLA_SUCCESS.
 */
SPLA_EXPORT SplaError spla_ctx_destroy(SplaContext* ctx);

/**
 * Access a Context parameter.
 * @param[in] ctx Context handle.
 * @param[out] pu Procesing unit used for computations.
 * @return Error code or SPLA_SUCCESS.
 */
SPLA_EXPORT SplaError spla_ctx_processing_unit(SplaContext ctx, SplaProcessingUnit* pu);

/**
 * Access a Context parameter.
 * @param[in] ctx Context handle.
 * @param[out] numThreads Maximum number of threads used for computations.
 * @return Error code or SPLA_SUCCESS.
 */
SPLA_EXPORT SPLA_DEPRECATED SplaError spla_ctx_num_threads(SplaContext ctx, int* numThreads);

/**
 * Access a Context parameter.
 * @param[in] ctx Context handle.
 * @param[out] numTiles Number of tiles used to overlap computation and communication.
 * @return Error code or SPLA_SUCCESS.
 */
SPLA_EXPORT SplaError spla_ctx_num_tiles(SplaContext ctx, int* numTiles);

/**
 * Access a Context parameter.
 * @param[in] ctx Context handle.
 * @param[out] tileSizeHost Size of tiles for host compuations and partitioning of communication.
 * @return Error code or SPLA_SUCCESS.
 */
SPLA_EXPORT SplaError spla_ctx_tile_size_host(SplaContext ctx, int* tileSizeHost);

/**
 * Access a Context parameter.
 * @param[in] ctx Context handle.
 * @param[out] tileSizeGPU Size of tiles on GPU.
 * @return Error code or SPLA_SUCCESS.
 */
SPLA_EXPORT SplaError spla_ctx_tile_size_gpu(SplaContext ctx, int* tileSizeGPU);

/**
 * Access a Context parameter.
 * @param[in] ctx Context handle.
 * @param[out] opThresholdGPU Operations threshold, below which computation may be done on Host,
 * even if processing unit is set to GPU. For GEMM, the number of operations is estimatex as 2mnk.
 * @return Error code or SPLA_SUCCESS.
 */
SPLA_EXPORT SplaError spla_ctx_op_threshold_gpu(SplaContext ctx, int* opThresholdGPU);

/**
 * Access a Context parameter.
 * @param[in] ctx Context handle.
 * @param[out] deviceId Id of GPU used for computations. This is set as fixed parameter by query of
 * device id at context creation.
 * @return Error code or SPLA_SUCCESS.
 */
SPLA_EXPORT SplaError spla_ctx_gpu_device_id(SplaContext ctx, int* deviceId);

/**
 * Access a Context parameter.
 * @param[in] ctx Context handle.
 * @param[in] size Total allocated memory on host in bytes used for internal buffers. Does not
 * include allocations through standard C++ allocators. May change with with use of context.
 * @return Error code or SPLA_SUCCESS.
 */
SPLA_EXPORT SplaError spla_ctx_allocated_memory_host(SplaContext ctx, uint_least64_t* size) ;

/**
 * Access a Context parameter.
 * @param[in] ctx Context handle.
 * @param[in] size Total allocated pinned memory on host in bytes used for internal buffers. Does
 * not include allocations through standard C++ allocators. May change with with use of context.
 * @return Error code or SPLA_SUCCESS.
 */
SPLA_EXPORT SplaError spla_ctx_allocated_memory_pinned(SplaContext ctx, uint_least64_t* size) ;

/**
 * Access a Context parameter.
 * @param[in] ctx Context handle.
 * @param[in] size Total allocated memory on gpu in bytes used for internal buffers. Does not
 * include allocations by device libraries like cuBLAS / rocBLAS. May change with with use of
 * context.
 * @return Error code or SPLA_SUCCESS.
 */
SPLA_EXPORT SplaError spla_ctx_allocated_memory_gpu(SplaContext ctx, uint_least64_t* size) ;

/**
 * Set the number of threads to be used.
 *
 * @param[in] ctx Context handle.
 * @param[in] numThreads Number of threads.
 * @return Error code or SPLA_SUCCESS.
 */
SPLA_EXPORT SPLA_DEPRECATED SplaError spla_ctx_set_num_threads(SplaContext ctx, int numThreads);

/**
 * Set the number of tiles.
 *
 * @param[in] ctx Context handle.
 * @param[in] numTiles Number of tiles.
 * @return Error code or SPLA_SUCCESS.
 */
SPLA_EXPORT SplaError spla_ctx_set_num_tiles(SplaContext ctx, int numTiles);

/**
 * Set the tile size used for computations on host and partitioning communication.
 *
 * @param[in] ctx Context handle.
 * @param[in] tileSizeHost Size of tiles on host.
 * @return Error code or SPLA_SUCCESS.
 */
SPLA_EXPORT SplaError spla_ctx_set_tile_size_host(SplaContext ctx, int tileSizeHost);

/**
 * Set the operations threshold, below which computation may be done on Host, even if processing
 * unit is set to GPU. For GEMM, the number of operations is estimatex as 2mnk.
 *
 * @param[in] ctx Context handle.
 * @param[in] opThresholdGPU Threshold in number of operations.
 * @return Error code or SPLA_SUCCESS.
 */
SPLA_EXPORT SplaError spla_ctx_set_op_threshold_gpu(SplaContext ctx, int opThresholdGPU);

/**
 * Set tile size for GPU computations.
 *
 * @param[in] ctx Context handle.
 * @param[in] tileSizeGPU Size of tiles on GPU.
 * @return Error code or SPLA_SUCCESS.
 */
SPLA_EXPORT SplaError spla_ctx_set_tile_size_gpu(SplaContext ctx, int tileSizeGPU);

/**
 * Set the allocation and deallocation functions for host memory. Internal default uses a memory
 * pool for better performance. Not available in Fortran module.
 *
 * @param[in] ctx Context handle.
 * @param[in] allocateFunc Function allocating given size in bytes.
 * @param[in] deallocateFunc Function to deallocate memory allocated using allocateFunc.
 * @return Error code or SPLA_SUCCESS.
 */
SPLA_EXPORT SplaError spla_ctx_set_alloc_host(SplaContext ctx, void* (*allocateFunc)(size_t),
                                              void (*deallocateFunc)(void*));

/**
 * Set the allocation and deallocation functions for pinned host memory. Internal default uses a
 * memory pool for better performance. Not available in Fortran module.
 *
 * @param[in] ctx Context handle.
 * @param[in] allocateFunc Function allocating given size in bytes.
 * @param[in] deallocateFunc Function to deallocate memory allocated using allocateFunc.
 * @return Error code or SPLA_SUCCESS.
 */
SPLA_EXPORT SplaError spla_ctx_set_alloc_pinned(SplaContext ctx, void* (*allocateFunc)(size_t),
                                                void (*deallocateFunc)(void*));

/**
 * Set the allocation and deallocation functions for gpu memory. Internal default uses a
 * memory pool for better performance. Not available in Fortran module.
 *
 * @param[in] ctx Context handle.
 * @param[in] allocateFunc Function allocating given size in bytes.
 * @param[in] deallocateFunc Function to deallocate memory allocated using allocateFunc.
 * @return Error code or SPLA_SUCCESS.
 */
SPLA_EXPORT SplaError spla_ctx_set_alloc_gpu(SplaContext ctx, void* (*allocateFunc)(size_t),
                                             void (*deallocateFunc)(void*));

#ifdef __cplusplus
}
#endif
#endif
