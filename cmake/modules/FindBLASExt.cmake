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
# FindBLASExt
# -----------
#
# This module tries to find the BLAS library (libsci) by Cray.
#
# The following variables are set
#
# ::
#
#   BLASExt_FOUND           - True if blas_ext is found
#   BLASExt_LIBRARIES       - The required libraries
#
# The following import target is created
#
# ::
#
#   BLAS::BLAS_EXT

# set paths to look for library from ROOT variables.If new policy is set, find_library() automatically uses them.
if(NOT POLICY CMP0074)
    set(_BLASExt_PATHS ${BLASExt_ROOT} $ENV{BLASExt_ROOT})
endif()

set(BLASExt_VENDOR "${BLA_VENDOR}" CACHE STRING "")

if(BLA_VENDOR AND NOT "${BLA_VENDOR}" STREQUAL "${BLASExt_VENDOR}")
    set(BLASExt_LIBRARIES "" CACHE STRING "" FORCE)
    set(BLASExt_VENDOR "${BLA_VENDOR}" CACHE STRING "" FORCE)
endif()

set(BLASExt_LIBRARIES_DEPS)
macro(find_blas_ext)

	if(BLA_VENDOR AND "${BLA_VENDOR}" STREQUAL "CRAY_LIBSCI")
        set(_sci_lib "sci_gnu")

        if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
            set(_sci_lib "sci_intel")
        elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
            set(_sci_lib "sci_cray")
        endif()

        # use multi-threaded version if OpenMP available
        find_package(OpenMP QUIET COMPONENTS CXX)
        if(TARGET OpenMP::OpenMP_CXX)
            set(_sci_lib ${_sci_lib}_mp ${_sci_lib})
        endif()

        find_library(
            BLASExt_LIBRARIES
            NAMES ${_sci_lib}
            HINTS ${_BLASExt_PATHS}
            ENV CRAY_LIBBLASExt_PREFIX_DIR
            PATH_SUFFIXES "lib" "lib64"
        )
        if(BLASExt_LIBRARIES AND TARGET OpenMP::OpenMP_CXX)
            list(APPEND BLASExt_LIBRARIES_DEPS $<LINK_ONLY:OpenMP::OpenMP_CXX>)
        endif()

	else()
        find_package(BLAS MODULE ${ARGV})
        if(BLAS_FOUND AND NOT BLASExt_LIBRARIES)
            message(STATUS "BLAS_LIBRARIES= \"${BLAS_LIBRARIES}\"")
            set(BLASExt_LIBRARIES "${BLAS_LIBRARIES} ${BLAS_LINKER_FLAGS}" CACHE STRING "" FORCE)
		endif()
	endif()
endmacro()

if(NOT BLA_VENDOR)
    set(_BLAS_VENDOR_LIST Intel10_64lp AOCL_mt Arm_mp OpenBLAS FLAME CRAY_LIBSCI)
	foreach(BLA_VENDOR IN LISTS _BLAS_VENDOR_LIST)
        if(NOT BLASExt_LIBRARIES)
            message(STATUS "Looking for ${BLA_VENDOR}")
			find_blas_ext(QUIET)
		endif()
	endforeach()
	# if not found, search for any BLAS library
	unset(BLA_VENDOR)
endif()
if(NOT BLASExt_LIBRARIES)
	find_blas_ext(REQUIRED)
endif()


# check if found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(BLASExt REQUIRED_VARS BLASExt_LIBRARIES)

# add target to link against
if(BLASExt_FOUND)
    if(NOT TARGET BLAS::blas_ext)
        add_library(BLAS::BLAS_EXT INTERFACE IMPORTED)
    endif()
    string(STRIP "${BLASExt_LIBRARIES} ${BLASExt_LIBRARIES_DEPS}" BLASExt_LIBRARIES_STRIPPED)
    set_property(TARGET BLAS::BLAS_EXT PROPERTY INTERFACE_LINK_LIBRARIES ${BLASExt_LIBRARIES_STRIPPED})
endif()

# prevent clutter in cache
MARK_AS_ADVANCED(BLASExt_FOUND BLASExt_LIBRARIES)
