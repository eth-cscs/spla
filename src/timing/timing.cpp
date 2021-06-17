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

#include "timing/timing.hpp"

#include "spla/config.h"
#include "spla/errors.h"
#include "spla/types.h"
#include "timing/rt_graph.hpp"

namespace spla {
namespace timing {
::rt_graph::Timer GlobalTimer;
}  // namespace timing
}  // namespace spla

#ifdef SPLA_TIMING
#include <fstream>
#include <iostream>
#include <string>

extern "C" {

SPLA_EXPORT SplaError spla_timer_start(int n, const char* name) {
  try {
    spla::timing::GlobalTimer.start(std::string(name, n));
    return SPLA_SUCCESS;
  } catch (...) {
    return SPLA_UNKNOWN_ERROR;
  }
}

SPLA_EXPORT SplaError spla_timer_stop(int n, const char* name) {
  try {
    spla::timing::GlobalTimer.stop(std::string(name, n));
    return SPLA_SUCCESS;
  } catch (...) {
    return SPLA_UNKNOWN_ERROR;
  }
}

SPLA_EXPORT SplaError spla_timer_export_json(int n, const char* name) {
  try {
    std::string fileName(name, n);
    std::ofstream file(fileName);
    file << spla::timing::GlobalTimer.process().json();
    return SPLA_SUCCESS;
  } catch (...) {
    return SPLA_UNKNOWN_ERROR;
  }
}

SPLA_EXPORT SplaError spla_timer_print() {
  try {
    std::cout << spla::timing::GlobalTimer.process().print();
    return SPLA_SUCCESS;
  } catch (...) {
    return SPLA_UNKNOWN_ERROR;
  }
}
}
#endif
