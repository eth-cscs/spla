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
# FindSCI
# -----------
#
# This module tries to find the SCI library (libsci) by Cray.
#
# The following variables are set
#
# ::
#
#   SCI_FOUND           - True if sci is found
#   SCI_LIBRARIES       - The required libraries
#   SCI_INCLUDE_DIRS    - The required include directory
#
# The following import target is created
#
# ::
#
#   SCI::sci

#set paths to look for library from ROOT variables.If new policy is set, find_library() automatically uses them.
if(NOT POLICY CMP0074)
    set(_SCI_PATHS ${SCI_ROOT} $ENV{SCI_ROOT})
endif()

find_library(
    SCI_LIBRARIES
    NAMES "sci" "sci_gnu" "sci_intel" "sci_cray"
    HINTS ${_SCI_PATHS}
    PATH_SUFFIXES "lib" "lib64"
)
find_path(
    SCI_INCLUDE_DIRS
    NAMES "cblas.h"
    HINTS ${_SCI_PATHS}
    PATH_SUFFIXES "include"
)

# check if found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SCI REQUIRED_VARS SCI_INCLUDE_DIRS SCI_LIBRARIES)

# add target to link against
if(SCI_FOUND)
    if(NOT TARGET SCI::sci)
        add_library(SCI::sci INTERFACE IMPORTED)
    endif()
    set_property(TARGET SCI::sci PROPERTY INTERFACE_LINK_LIBRARIES ${SCI_LIBRARIES})
    set_property(TARGET SCI::sci PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${SCI_INCLUDE_DIRS})
endif()

# prevent clutter in cache
MARK_AS_ADVANCED(SCI_FOUND SCI_LIBRARIES SCI_INCLUDE_DIRS)
