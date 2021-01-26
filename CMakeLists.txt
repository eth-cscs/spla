cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(spla LANGUAGES CXX VERSION 1.2.1)

# allow {module}_ROOT variables to be set
if(POLICY CMP0074)
	cmake_policy(SET CMP0074 NEW)
endif()

# use INTERFACE_LINK_LIBRARIES property if available
if(POLICY CMP0022)
	cmake_policy(SET CMP0022 NEW)
endif()

# set default build type to RELEASE
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
		"Debug" "Release" "MinSizeRel" "RelWithDebInfo"
		)
endif()

# set language and standard
set(CMAKE_CXX_STANDARD 11)

# Get GNU standard install prefixes
include(GNUInstallDirs)

#add local module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake/modules)

# Options
option(SPLA_STATIC "Compile as static library" OFF)
option(SPLA_OMP "Compile with OpenMP support" ON)
option(SPLA_BUILD_TESTS "Build tests" OFF)
option(SPLA_INSTALL "Enable CMake install commands" ON)

set(SPLA_GPU_BACKEND "OFF" CACHE STRING "GPU backend")
set_property(CACHE SPLA_GPU_BACKEND PROPERTY STRINGS
	"OFF" "CUDA" "ROCM"
	)
set(_SPLA_HOST_BLAS_LIST "AUTO" "MKL" "OPENBLAS" "BLIS" "CRAY_LIBSCI" "GENERIC")
set(SPLA_HOST_BLAS "AUTO" CACHE STRING "Blas library for computations on host")
set_property(CACHE SPLA_HOST_BLAS PROPERTY STRINGS ${_SPLA_HOST_BLAS_LIST})


# Options combination check
set(SPLA_CUDA OFF)
set(SPLA_ROCM OFF)
if(SPLA_GPU_BACKEND)
	if(SPLA_GPU_BACKEND STREQUAL "CUDA")
		set(SPLA_CUDA ON)
	elseif(SPLA_GPU_BACKEND STREQUAL "ROCM")
		set(SPLA_ROCM ON)
	else()
		message(FATAL_ERROR "Invalid GPU backend")
	endif()
endif()

if(NOT ${SPLA_HOST_BLAS} IN_LIST _SPLA_HOST_BLAS_LIST)
	message(FATAL_ERROR "Invalid Host BLAS backend")
endif()

# set preferred library type
if (SPLA_STATIC)
	set(SPLA_LIBRARY_TYPE STATIC)
else()
	set(SPLA_LIBRARY_TYPE SHARED)
endif()

set(SPLA_DEFINITIONS)
set(SPLA_EXTERNAL_COMPILE_OPTIONS)
set(SPLA_EXTERNAL_LIBS)
set(SPLA_INCLUDE_DIRS)
set(SPLA_EXTERNAL_INCLUDE_DIRS)
set(SPLA_EXTERNAL_PKG_PACKAGES)
set(SPLA_BLAS_OPENBLAS OFF)
set(SPLA_BLAS_MKL OFF)
set(SPLA_BLAS_BLIS OFF)
set(SPLA_BLAS_SCI OFF)
set(SPLA_BLAS_ATLAS OFF)
set(SPLA_RESTRICT_ATTR " ")

# Hide symbols by default if tests are not build
if(NOT SPLA_BUILD_TESTS)
	set(CMAKE_CXX_VISIBILITY_PRESET hidden)
	set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)
else()
	# disable visibility macros if tests are build
	list(APPEND SPLA_DEFINITIONS -DSPLA_STATIC_DEFINE)
	set(CMAKE_CXX_VISIBILITY_PRESET default)
	set(CMAKE_VISIBILITY_INLINES_HIDDEN 0)
endif()

# check if restrict attribute is available
include(CheckCXXSourceCompiles)
check_cxx_source_compiles(
	"
			int f(double *__restrict__ x);
			int main(void) {return 0;}
	"
	HAVE___RESTRICT__
	)
if(${HAVE___RESTRICT__})
	set(SPLA_RESTRICT_ATTR "__restrict__")
else()
	check_cxx_source_compiles(
			"
					int f(double *__restrict x);
					int main(void) {return 0;}
			"
			HAVE___RESTRICT
		)
	if(HAVE___RESTRICT)
		set(SPLA_RESTRICT_ATTR "__restrict")
	else()
		check_cxx_source_compiles(
			"
					int f(double *restrict x);
					int main(void) {return 0;}
			"
			HAVE_RESTRICT
			)
		if(${HAVE_RESTRICT})
			set(SPLA_RESTRICT_ATTR "restrict")
		endif()
	endif()
endif()


# MPI is always required
find_package(MPI COMPONENTS CXX REQUIRED)
list(APPEND SPLA_EXTERNAL_LIBS MPI::MPI_CXX)


if(SPLA_OMP)
	find_package(OpenMP REQUIRED)
	list(APPEND SPLA_EXTERNAL_LIBS OpenMP::OpenMP_CXX)
endif()

# CUDA
if(SPLA_CUDA)
	enable_language(CUDA)
	find_library(CUDA_CUDART_LIBRARY cudart PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
	find_library(CUDA_BLAS_LIBRARY cublas PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
	list(APPEND SPLA_EXTERNAL_LIBS ${CUDA_CUDART_LIBRARY} ${CUDA_BLAS_LIBRARY})
	list(APPEND SPLA_EXTERNAL_INCLUDE_DIRS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

# ROCm
if(SPLA_ROCM)
  find_package(ROCBLAS REQUIRED)
  find_package(HIPLIBS REQUIRED)
  list(APPEND SPLA_EXTERNAL_LIBS ROCBLAS::rocblas HIPLIBS::hiplibs)
  list(APPEND SPLA_DEFINITIONS -D__HIP_PLATFORM_HCC__) # required for parsing hip headers
endif()

# find BLAS backend for Host computation.
set(_SPLA_BLAS_FOUND FALSE)
set(_SPLA_FIND_BLAS_OPTION)
if(NOT ${SPLA_HOST_BLAS} STREQUAL "AUTO")
	set(_SPLA_FIND_BLAS_OPTION REQUIRED)
endif()


if(${SPLA_HOST_BLAS} STREQUAL "AUTO" OR ${SPLA_HOST_BLAS} STREQUAL "MKL")
	find_package(MKL QUIET ${_SPLA_FIND_BLAS_OPTION})
	if(SPLA_BUILD_TESTS)
		set(SPLA_MKL_TARGET mkl::scalapack_mpich_intel_32bit_omp_dyn)
	else()
		set(SPLA_MKL_TARGET mkl::mkl_intel_32bit_omp_dyn)
	endif()
	if(TARGET ${SPLA_MKL_TARGET})
		message(STATUS "Host BLAS Backend: MKL")
		list(APPEND SPLA_EXTERNAL_LIBS ${SPLA_MKL_TARGET})
		set(SPLA_BLAS_MKL ON)
		set(SPLA_BLAS_HEADER_NAME mkl.h)
		set(_SPLA_BLAS_FOUND TRUE)
	endif()
endif()

if(NOT ${_SPLA_BLAS_FOUND} AND (${SPLA_HOST_BLAS} STREQUAL "AUTO" OR ${SPLA_HOST_BLAS} STREQUAL "BLIS"))
	find_package(BLIS ${_SPLA_FIND_BLAS_OPTION})
	if(TARGET BLIS::blis)
		message(STATUS "Host BLAS Backend: BLIS")
		list(APPEND SPLA_EXTERNAL_LIBS BLIS::blis)
		set(SPLA_BLAS_BLIS ON)
		set(SPLA_BLAS_HEADER_NAME cblas.h)
		set(_SPLA_BLAS_FOUND TRUE)
	endif()
endif()

if(NOT ${_SPLA_BLAS_FOUND} AND (${SPLA_HOST_BLAS} STREQUAL "AUTO" OR ${SPLA_HOST_BLAS} STREQUAL "OPENBLAS"))
	find_package(OPENBLAS ${_SPLA_FIND_BLAS_OPTION})
	if(TARGET OPENBLAS::openblas)
		message(STATUS "Host BLAS Backend: OPENBLAS")
		list(APPEND SPLA_EXTERNAL_LIBS OPENBLAS::openblas)
		set(SPLA_BLAS_OPENBLAS ON)
		set(_SPLA_BLAS_FOUND TRUE)
		# try to find openblas header file
		find_file(_BLAS_HEADER NAMES cblas_openblas.h cblas-openblas.h cblas.h HINTS ${OPENBLAS_INCLUDE_DIRS})
		if(_BLAS_HEADER)
			get_filename_component(SPLA_BLAS_HEADER_NAME ${_BLAS_HEADER} NAME)
		endif()
	endif()
endif()

if(NOT ${_SPLA_BLAS_FOUND} AND (${SPLA_HOST_BLAS} STREQUAL "AUTO" OR ${SPLA_HOST_BLAS} STREQUAL "CRAY_LIBSCI"))
	find_package(SCI ${_SPLA_FIND_BLAS_OPTION})
	if(TARGET SCI::sci)
		message(STATUS "Host BLAS Backend: CRAY_LIBSCI")
		list(APPEND SPLA_EXTERNAL_LIBS SCI::sci)
		set(SPLA_BLAS_SCI ON)
		set(SPLA_BLAS_HEADER_NAME cblas.h)
		set(_SPLA_BLAS_FOUND TRUE)
	endif()
endif()

if(NOT ${_SPLA_BLAS_FOUND} AND (${SPLA_HOST_BLAS} STREQUAL "AUTO" OR ${SPLA_HOST_BLAS} STREQUAL "ATLAS"))
	find_package(ATLAS ${_SPLA_FIND_BLAS_OPTION})
	if(TARGET ATLAS::atlas)
		message(STATUS "Host BLAS Backend: ATLAS")
		list(APPEND SPLA_EXTERNAL_LIBS ATLAS::atlas)
		set(SPLA_BLAS_ATLAS ON)
		set(_SPLA_BLAS_FOUND TRUE)
		# try to find header file
		find_file(_BLAS_HEADER NAMES cblas_atlas.h cblas-atlas.h cblas.h HINTS ${ATLAS_INCLUDE_DIRS})
		if(_BLAS_HEADER)
			get_filename_component(SPLA_BLAS_HEADER_NAME ${_BLAS_HEADER} NAME)
		endif()
	endif()
endif()

if(NOT ${_SPLA_BLAS_FOUND} AND (${SPLA_HOST_BLAS} STREQUAL "AUTO" OR ${SPLA_HOST_BLAS} STREQUAL "GENERIC"))
	find_package(GenericBLAS ${_SPLA_FIND_BLAS_OPTION})
	if(TARGET GenericBLAS::blas)
		message(STATUS "Host BLAS Backend: GENERIC")
		message(STATUS "Host BLAS libs: ${GenericBLAS_LIBRARIES}")
		list(APPEND SPLA_EXTERNAL_LIBS GenericBLAS::blas)
		set(_SPLA_BLAS_FOUND TRUE)
		# try to find header file
		find_file(_BLAS_HEADER NAMES cblas.h HINTS ${GenericBLAS_INCLUDE_DIRS})
		if(_BLAS_HEADER)
			get_filename_component(SPLA_BLAS_HEADER_NAME ${_BLAS_HEADER} NAME)
		endif()
	endif()
endif()

# Fall back to CMake provided FindBLAS as last resort or for GENERIC blas library
if(NOT _SPLA_BLAS_FOUND)
	find_package(BLAS REQUIRED)
	message(STATUS "Host BLAS Backend: ${BLAS_LIBRARIES}")
	find_file(_BLAS_HEADER NAMES cblas.h)
	if(_BLAS_HEADER)
		get_filename_component(SPLA_BLAS_HEADER_NAME ${_BLAS_HEADER} NAME)
	endif()
	set(_SPLA_BLAS_FOUND TRUE)
	if(NOT TARGET BLAS::blas)
		add_library(BLAS::blas INTERFACE IMPORTED)
	endif()
	set_property(TARGET BLAS::blas PROPERTY INTERFACE_LINK_LIBRARIES ${BLAS_LIBRARIES} ${BLAS_LINKER_FLAGS})
	list(APPEND SPLA_EXTERNAL_LIBS BLAS::blas)
endif()


# make sure cblas symbols exist in blas library
include(CheckCXXSymbolExists)
set(CMAKE_REQUIRED_LIBRARIES ${SPLA_EXTERNAL_LIBS})
unset(_SPLA_CBLAS_FOUND CACHE) # Result is cached, so change of library will not lead to a new check automatically
if(SPLA_BLAS_HEADER_NAME)
	check_cxx_symbol_exists(cblas_dgemm ${SPLA_BLAS_HEADER_NAME} _SPLA_CBLAS_FOUND)
else()
	set(CMAKE_REQUIRED_INCLUDES ${PROJECT_SOURCE_DIR}/cmake/util)
	check_cxx_symbol_exists(cblas_dgemm blas_dgemm_symbol.h _SPLA_CBLAS_FOUND)
endif()
if(NOT _SPLA_CBLAS_FOUND)
	message(FATAL_ERROR "CBlas symbols are required but not found in blas library!")
endif()


if(SPLA_BUILD_TESTS)
	# enable timing with testing
	set(SPLA_TIMING ON)
else()
	set(SPLA_TIMING OFF)
endif()

# generate config.h
configure_file(include/spla/config.h.in ${PROJECT_BINARY_DIR}/spla/config.h)

list(APPEND SPLA_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/src)
list(APPEND SPLA_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include)
list(APPEND SPLA_INCLUDE_DIRS ${PROJECT_BINARY_DIR})
list(APPEND SPLA_EXTERNAL_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/ext)

#############################################################################
# All include dirs and definitions must be set before sub-directory is added!
#############################################################################
add_subdirectory(src)

# add tests for developement
if(SPLA_BUILD_TESTS)
	add_subdirectory(tests)
endif()
