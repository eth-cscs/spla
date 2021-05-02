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
#ifndef SPLA_GPU_MATRIX_ACCESSOR_HPP
#define SPLA_GPU_MATRIX_ACCESSOR_HPP

#include "spla/config.h"
#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
#include <cassert>
#include <memory>

#include "gpu_util/gpu_blas_api.hpp"
#include "gpu_util/gpu_runtime_api.hpp"
#include "gpu_util/gpu_stream_handle.hpp"
#include "gpu_util/gpu_transfer.hpp"
#include "memory/buffer.hpp"
#include "memory/gpu_allocator.hpp"
#include "memory/gpu_array_const_view.hpp"
#include "memory/gpu_array_view.hpp"
#include "memory/host_array_const_view.hpp"
#include "spla/exceptions.hpp"
#include "util/common_types.hpp"

namespace spla {

template <typename T>
class GPUMatrixAccessor {
public:
  using ValueType = T;

  GPUMatrixAccessor(const GPUArrayView2D<ValueType> &matrix)
      : matrixGPU_(matrix),
        rows_(matrix.dim_inner()),
        cols_(matrix.dim_outer()),
        maxTileSize_(matrix.size()) {}

  GPUMatrixAccessor(const HostArrayView2D<ValueType> &matrix, IntType maxTileSize,
                    std::shared_ptr<Buffer<GPUAllocator>> buffer)
      : matrixHost_(matrix),
        rows_(matrix.dim_inner()),
        cols_(matrix.dim_outer()),
        maxTileSize_(maxTileSize),
        buffer_(std::move(buffer)) {
    assert(buffer_);
  }

  auto max_tile_size() const -> IntType { return maxTileSize_; }

  auto rows() const -> IntType { return rows_; }

  auto cols() const -> IntType { return cols_; }

  auto size() const -> IntType { return rows_ * cols_; }

  auto copy_back(GPUArrayView2D<ValueType> view, IntType rowOffset, IntType colOffset,
                 const gpu::StreamType &stream) -> void {
    if (matrixGPU_.empty()) {
      assert(buffer_->data<ValueType>() <= view.data() &&
             buffer_->data<ValueType>() + buffer_->size<ValueType>() >=
                 view.data() + view.ld_inner() * view.dim_outer());
      copy_from_gpu_async(
          stream, GPUArrayConstView2D<ValueType>(view),
          HostArrayView2D<ValueType>(&matrixHost_(colOffset, rowOffset), view.dim_outer(),
                                     view.dim_inner(), matrixHost_.ld_inner()));
    }
  }

  auto get_tile(IntType rowOffset, IntType colOffset, IntType rows, IntType cols,
                const gpu::StreamType &stream) -> GPUArrayView2D<ValueType> {
    assert(rowOffset + rows <= rows_);
    assert(colOffset + cols <= cols_);
    assert(rows * cols <= maxTileSize_);
    if (matrixGPU_.empty()) {
      // grow buffer to twice the current required size to avoid reallocation for small size
      // increases
      if (buffer_->size<ValueType>() < cols * rows) {
        buffer_->resize<ValueType>(std::min<IntType>(2 * cols * rows, maxTileSize_));
      }
      assert(buffer_->size<ValueType>() >= cols * rows);
      GPUArrayView2D<ValueType> tile(buffer_->data<ValueType>(), cols, rows);
      copy_to_gpu_async(stream,
                        HostArrayConstView2D<ValueType>(&matrixHost_(colOffset, rowOffset), cols,
                                                        rows, matrixHost_.ld_inner()),
                        tile);
      return tile;
    } else {
      return GPUArrayView2D<ValueType>(
          const_cast<ValueType *>(matrixGPU_.data()) + matrixGPU_.index(colOffset, rowOffset), cols,
          rows, matrixGPU_.ld_inner());
    }
  }

  auto sub_accessor(IntType rowOffset, IntType colOffset, IntType rows, IntType cols)
      -> GPUMatrixAccessor<ValueType> {
    assert(rowOffset + rows <= rows_);
    assert(colOffset + cols <= cols_);
    if (matrixGPU_.empty()) {
      return GPUMatrixAccessor<ValueType>(
          HostArrayView2D<ValueType>((matrixHost_.data() + matrixHost_.index(colOffset, rowOffset)),
                                     cols, rows, matrixHost_.ld_inner()),
          maxTileSize_, buffer_);
    } else {
      return GPUMatrixAccessor<ValueType>(
          GPUArrayView2D<ValueType>(matrixGPU_.data() + matrixGPU_.index(colOffset, rowOffset),
                                    cols, rows, matrixGPU_.ld_inner()));
    }
  }

private:
  HostArrayView2D<ValueType> matrixHost_;
  GPUArrayView2D<ValueType> matrixGPU_;
  IntType rows_, cols_;
  IntType maxTileSize_;
  std::shared_ptr<Buffer<GPUAllocator>> buffer_;
};

template <typename T>
class GPUConstMatrixAccessor {
public:
  using ValueType = T;

  GPUConstMatrixAccessor(const GPUArrayConstView2D<ValueType> &matrix)
      : matrixGPU_(matrix),
        rows_(matrix.dim_inner()),
        cols_(matrix.dim_outer()),
        maxTileSize_(matrix.size()) {}

  GPUConstMatrixAccessor(const HostArrayConstView2D<ValueType> &matrix, IntType maxTileSize,
                         std::shared_ptr<Buffer<GPUAllocator>> buffer)
      : matrixHost_(matrix),
        rows_(matrix.dim_inner()),
        cols_(matrix.dim_outer()),
        maxTileSize_(maxTileSize),
        buffer_(std::move(buffer)),
        currentOrigin_(new HostArrayConstView2D<T>()) {
    assert(buffer_);
  }

  auto max_tile_size() const -> IntType { return maxTileSize_; }

  auto rows() const -> IntType { return rows_; }

  auto cols() const -> IntType { return cols_; }

  auto size() const -> IntType { return rows_ * cols_; }

  auto get_tile(IntType rowOffset, IntType colOffset, IntType rows, IntType cols,
                const gpu::StreamType &stream) const -> GPUArrayConstView2D<ValueType> {
    assert(rowOffset + rows <= rows_);
    assert(colOffset + cols <= cols_);
    assert(rows * cols <= maxTileSize_);
    if (matrixGPU_.empty()) {
      // grow buffer to twice the current required size to avoid reallocation for small size
      // increases
      if (buffer_->size<ValueType>() < cols * rows) {
        buffer_->resize<ValueType>(std::min<IntType>(2 * cols * rows, maxTileSize_));
      }
      assert(buffer_->size<ValueType>() >= cols * rows);
      GPUArrayView2D<ValueType> tile(buffer_->data<ValueType>(), cols, rows);

      HostArrayConstView2D<ValueType> tileHost(&matrixHost_(colOffset, rowOffset), cols, rows,
                                               matrixHost_.ld_inner());

      // Only copy if not already in device memory
      if (!currentOrigin_ || *currentOrigin_ != tileHost) {
        copy_to_gpu_async(stream, tileHost, tile);
      }

      *currentOrigin_ = tileHost;

      return tile;
    } else {
      return GPUArrayConstView2D<ValueType>(
          matrixGPU_.data() + matrixGPU_.index(colOffset, rowOffset), cols, rows,
          matrixGPU_.ld_inner());
    }
  }

  auto sub_accessor(IntType rowOffset, IntType colOffset, IntType rows, IntType cols) const
      -> GPUConstMatrixAccessor<ValueType> {
    assert(rowOffset + rows <= rows_);
    assert(colOffset + cols <= cols_);
    if (matrixGPU_.empty()) {
      return GPUConstMatrixAccessor<ValueType>(
          HostArrayConstView2D<ValueType>(
              (matrixHost_.data() + matrixHost_.index(colOffset, rowOffset)), cols, rows,
              matrixHost_.ld_inner()),
          maxTileSize_, buffer_, currentOrigin_);
    } else {
      return GPUConstMatrixAccessor<ValueType>(
          GPUArrayConstView2D<ValueType>(matrixGPU_.data() + matrixGPU_.index(colOffset, rowOffset),
                                         cols, rows, matrixGPU_.ld_inner()));
    }
  }

private:
  GPUConstMatrixAccessor(const HostArrayConstView2D<ValueType> &matrix, IntType maxTileSize,
                         std::shared_ptr<Buffer<GPUAllocator>> buffer,
                         std::shared_ptr<HostArrayConstView2D<ValueType>> currentOrigin)
      : matrixHost_(matrix),
        rows_(matrix.dim_inner()),
        cols_(matrix.dim_outer()),
        maxTileSize_(maxTileSize),
        buffer_(std::move(buffer)),
        currentOrigin_(std::move(currentOrigin)) {
    assert(buffer_);
    assert(currentOrigin_);
  }

  HostArrayConstView2D<ValueType> matrixHost_;
  GPUArrayConstView2D<ValueType> matrixGPU_;
  IntType rows_, cols_;
  IntType maxTileSize_;
  std::shared_ptr<Buffer<GPUAllocator>> buffer_;
  std::shared_ptr<HostArrayConstView2D<ValueType>> currentOrigin_;
};

}  // namespace spla

#endif
#endif
