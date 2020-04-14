#  Copyright (c) 2019 ETH Zurich, Simon Frasch
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#  2. Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.


#.rst:
# FindSCALAPACK
# -----------
#
# This module searches for the ScaLAPACK library.
#
# The following variables are set
#
# ::
#
#   SCALAPACK_FOUND           - True if double precision ScaLAPACK library is found
#   SCALAPACK_FLOAT_FOUND     - True if single precision ScaLAPACK library is found
#   SCALAPACK_LIBRARIES       - The required libraries
#   SCALAPACK_INCLUDE_DIRS    - The required include directory
#
# The following import target is created
#
# ::
#
#   SCALAPACK::SCALAPACK



# set paths to look for library
set(_SCALAPACK_PATHS ${SCALAPACK_ROOT} $ENV{SCALAPACK_ROOT})
set(_SCALAPACK_INCLUDE_PATHS)

set(_SCALAPACK_DEFAULT_PATH_SWITCH)

if(_SCALAPACK_PATHS)
    # disable default paths if ROOT is set
    set(_SCALAPACK_DEFAULT_PATH_SWITCH NO_DEFAULT_PATH)
else()
    # try to detect location with pkgconfig
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
      pkg_check_modules(PKG_SCALAPACK QUIET "scalapack")
    endif()
    set(_SCALAPACK_PATHS ${PKG_SCALAPACK_LIBRARY_DIRS})
    set(_SCALAPACK_INCLUDE_PATHS ${PKG_SCALAPACK_INCLUDE_DIRS})
endif()

find_library(
    SCALAPACK_LIBRARIES
    NAMES "scalapack" "scalapack-mpich" "scalapack-openmpi"
    HINTS ${_SCALAPACK_PATHS}
    PATH_SUFFIXES "lib" "lib64"
    ${_SCALAPACK_DEFAULT_PATH_SWITCH}
)

# check if found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SCALAPACK REQUIRED_VARS SCALAPACK_LIBRARIES )


# add target to link against
if(SCALAPACK_FOUND)
    if(NOT TARGET SCALAPACK::SCALAPACK)
        add_library(SCALAPACK::SCALAPACK INTERFACE IMPORTED)
    endif()
    set_property(TARGET SCALAPACK::SCALAPACK PROPERTY INTERFACE_LINK_LIBRARIES ${SCALAPACK_LIBRARIES})
endif()

# prevent clutter in cache
MARK_AS_ADVANCED(SCALAPACK_FOUND SCALAPACK_LIBRARIES SCALAPACK_INCLUDE_DIRS pkgcfg_lib_PKG_SCALAPACK_scalapack )
