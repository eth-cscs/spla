#  Copyright (c) 2019 ETH Zurich
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
# FindOPENBLAS
# -----------
#
# This module tries to find the OPENBLAS library.
#
# The following variables are set
#
# ::
#
#   OPENBLAS_FOUND           - True if openblas is found
#   OPENBLAS_LIBRARIES       - The required libraries
#   OPENBLAS_INCLUDE_DIRS    - The required include directory
#
# The following import target is created
#
# ::
#
#   OPENBLAS::openblas

#set paths to look for library from ROOT variables.If new policy is set, find_library() automatically uses them.
if(NOT POLICY CMP0074)
    set(_OPENBLAS_PATHS ${OPENBLAS_ROOT} $ENV{OPENBLAS_ROOT})
endif()

find_library(
    OPENBLAS_LIBRARIES
    NAMES "openblas"
    HINTS ${_OPENBLAS_PATHS}
    PATH_SUFFIXES "openblas/lib" "openblas/lib64" "openblas"
)
find_path(
    OPENBLAS_INCLUDE_DIRS
    NAMES "cblas-openblas.h" "cblas_openblas.h" "cblas.h" 
    HINTS ${_OPENBLAS_PATHS}
    PATH_SUFFIXES "openblas" "openblas/include" "include/openblas"
)

# check if found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OPENBLAS REQUIRED_VARS OPENBLAS_INCLUDE_DIRS OPENBLAS_LIBRARIES)

# add target to link against
if(OPENBLAS_FOUND)
    if(NOT TARGET OPENBLAS::openblas)
        add_library(OPENBLAS::openblas INTERFACE IMPORTED)
    endif()
    set_property(TARGET OPENBLAS::openblas PROPERTY INTERFACE_LINK_LIBRARIES ${OPENBLAS_LIBRARIES})
    set_property(TARGET OPENBLAS::openblas PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${OPENBLAS_INCLUDE_DIRS})
endif()

# prevent clutter in cache
MARK_AS_ADVANCED(OPENBLAS_FOUND OPENBLAS_LIBRARIES OPENBLAS_INCLUDE_DIRS)
