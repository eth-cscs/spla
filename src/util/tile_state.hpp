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
#ifndef SPLA_TILE_STATE_HPP
#define SPLA_TILE_STATE_HPP

#include <atomic>
#include "spla/config.h"
namespace spla {
enum class TileState : int {
  Empty,
  Prepared,
  PartiallyProcessed,
  Multiplied,
  InExchange,
  Exchanged
};

class AtomicTileState {
public:
  inline AtomicTileState(TileState state) noexcept : state_(state) {}

  inline AtomicTileState(const AtomicTileState& state) noexcept : state_(state.get()) {}

  inline AtomicTileState(AtomicTileState&& state) noexcept { this->set(state.get()); };

  inline auto operator=(const AtomicTileState& state) noexcept -> AtomicTileState& {
    this->set(state.get());
    return *this;
  }

  inline auto operator=(AtomicTileState&& state) noexcept -> AtomicTileState& {
    this->set(state.get());
    return *this;
  }

  inline auto get() const noexcept -> TileState { return state_.load(std::memory_order_acquire); }

  inline auto set(TileState state) noexcept -> void {
    state_.store(state, std::memory_order_release);
  }

private:
  std::atomic<TileState> state_;
};
}  // namespace spla
#endif
