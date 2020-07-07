cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(spla LANGUAGES CXX VERSION 0.1.0)

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


# Options combination check
set(SPLA_CUDA OFF)
set(SPLA_ROCM OFF)
if(SPLA_GPU_BACKEND)
	if(SPLA_GPU_BACKEND STREQUAL "CUDA")
		set(SPLA_CUDA ON)
	elseif(SPLA_GPU_BACKEND STREQUAL "ROCM")
		set(SPLA_ROCM ON)
	else()
		message(FATAL_ERROR "Invalid GPU backend option")
	endif()
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


# MPI is always required
find_package(MPI REQUIRED)
list(APPEND SPLA_EXTERNAL_LIBS MPI::MPI_CXX)

# find BLAS backend for Host computation. Prefer MKL if available.
find_package(MKL QUIET)

if(SPLA_BUILD_TESTS)
	set(SPLA_MKL_TARGET mkl::scalapack_mpich_intel_32bit_seq_st)
else()
	set(SPLA_MKL_TARGET mkl::mkl_intel_32bit_seq_st)
endif()

if(TARGET ${SPLA_MKL_TARGET})
	message(STATUS "BLAS Backend: MKL")
	list(APPEND SPLA_EXTERNAL_LIBS ${SPLA_MKL_TARGET})
	set(SPLA_BLAS_MKL ON)
	set(SPLA_BLAS_HEADER_NAME mkl.h)
else()
	# try to find OpenBLAS first
	set(BLA_VENDOR OpenBLAS)
	find_package(BLAS QUIET)
	if(BLAS_FOUND)
		message(STATUS "BLAS Backend: OpenBLAS")
		set(SPLA_BLAS_OPENBLAS ON)

		# try to find openblas header file
		find_file(_OPENBLAS_HEADER NAMES cblas_openblas.h cblas-openblas.h cblas.h)
		if(_OPENBLAS_HEADER)
			get_filename_component(SPLA_BLAS_HEADER_NAME ${_OPENBLAS_HEADER} NAME)
			get_filename_component(BLAS_INCLUDE_DIRS ${_OPENBLAS_HEADER} DIRECTORY)
		endif()

	else()
		# Fall back to generic generic blas implementation
		unset(BLA_VENDOR)
		unset(SPLA_BLAS_HEADER_NAME)
		find_package(BLAS REQUIRED)
		message(STATUS "BLAS Backend: GENERIC")
	endif()

	if(NOT TARGET BLAS::blas)
		add_library(BLAS::blas INTERFACE IMPORTED)
		set_property(TARGET BLAS::blas PROPERTY INTERFACE_LINK_LIBRARIES ${BLAS_LIBRARIES} ${BLAS_LINKER_FLAGS})
		if(BLAS_INCLUDE_DIRS)
			set_property(TARGET BLAS::blas PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${BLAS_INCLUDE_DIRS})
		endif()
	endif()
	list(APPEND SPLA_EXTERNAL_LIBS BLAS::blas)
endif()

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

# reset cmake library suffixes
if(SPLA_STATIC)
	set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES_SAVE})
endif()
