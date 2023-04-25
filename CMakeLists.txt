cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(spla LANGUAGES CXX VERSION 1.5.5)
set(SPLA_SO_VERSION 1)
set(SPLA_VERSION ${PROJECT_VERSION})

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
option(SPLA_BUILD_EXAMPLES "Compile examples" OFF)
option(SPLA_INSTALL "Enable CMake install commands" ON)
option(SPLA_FORTRAN "Compile fortran module" OFF)

set(SPLA_GPU_BACKEND "OFF" CACHE STRING "GPU backend")
set_property(CACHE SPLA_GPU_BACKEND PROPERTY STRINGS
	"OFF" "CUDA" "ROCM"
	)
set(_SPLA_HOST_BLAS_LIST "AUTO" "MKL" "ARMPL" "OPENBLAS" "BLIS" "CRAY_LIBSCI" "GENERIC")
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

set(SPLA_BLAS_MKL OFF)
set(SPLA_BLAS_BLIS OFF)
set(SPLA_BLAS_OPENBLAS OFF)
set(SPLA_BLAS_SCI OFF)
set(SPLA_BLAS_ATLAS OFF)
set(SPLA_BLAS_GENERIC OFF)
set(SPLA_BLAS_UNKNOWN OFF)

# Fortran
if(SPLA_FORTRAN)
	enable_language(Fortran)
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
set(SPLA_BLAS_ARMPL OFF)
set(SPLA_BLAS_BLIS OFF)
set(SPLA_BLAS_SCI OFF)
set(SPLA_BLAS_ATLAS OFF)
set(SPLA_RESTRICT_ATTR " ")

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
	find_package(OpenMP REQUIRED COMPONENTS CXX)
	list(APPEND SPLA_EXTERNAL_LIBS OpenMP::OpenMP_CXX)
endif()

# CUDA
if(SPLA_CUDA)
	if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.17.0") 
		find_package(CUDAToolkit REQUIRED)
	else()
		enable_language(CUDA)
		find_library(CUDA_CUDART_LIBRARY cudart PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
		if(NOT TARGET CUDA::cudart)
			add_library(CUDA::cudart INTERFACE IMPORTED)
		endif()
		set_property(TARGET CUDA::cudart PROPERTY INTERFACE_LINK_LIBRARIES ${CUDA_CUDART_LIBRARY})
		set_property(TARGET CUDA::cudart PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

		find_library(CUDA_CUBLAS_LIBRARY cublas PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
		if(NOT TARGET CUDA::cublas)
			add_library(CUDA::cublas INTERFACE IMPORTED)
		endif()
		set_property(TARGET CUDA::cublas PROPERTY INTERFACE_LINK_LIBRARIES ${CUDA_CUBLAS_LIBRARY})
		set_property(TARGET CUDA::cublas PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
	endif()

	list(APPEND SPLA_EXTERNAL_LIBS CUDA::cudart CUDA::cublas)
endif()

# ROCm
if(SPLA_ROCM)
	find_package(hip CONFIG REQUIRED)
	find_package(rocblas CONFIG REQUIRED)
  list(APPEND SPLA_EXTERNAL_LIBS hip::host roc::rocblas)
endif()

# find BLAS backend for Host computation.
set(_SPLA_BLAS_FOUND FALSE)

if(${SPLA_HOST_BLAS} STREQUAL "AUTO" OR ${SPLA_HOST_BLAS} STREQUAL "MKL")
	find_package(MKL QUIET)
	if(SPLA_OMP)
		if(TARGET mkl::intel_32bit_omp_dyn)
			set(SPLA_MKL_TARGET intel_32bit_omp_dyn)
		elseif(TARGET mkl::intel_32bit_omp_st)
			set(SPLA_MKL_TARGET intel_32bit_omp_st)
		endif()
	else()
		if(TARGET mkl::intel_32bit_seq_dyn)
			set(SPLA_MKL_TARGET intel_32bit_seq_dyn)
		elseif(TARGET mkl::intel_32bit_seq_st)
			set(SPLA_MKL_TARGET intel_32bit_seq_st)
		endif()
	endif()
	if(TARGET mkl::${SPLA_MKL_TARGET})
		message(STATUS "Host BLAS Backend: MKL")
		list(APPEND SPLA_EXTERNAL_LIBS mkl::${SPLA_MKL_TARGET})
		set(SPLA_BLAS_MKL ON)
		set(SPLA_BLAS_HEADER_NAME mkl.h)
		set(_SPLA_BLAS_FOUND TRUE)
	endif()
endif()

if(NOT ${_SPLA_BLAS_FOUND} AND (${SPLA_HOST_BLAS} STREQUAL "AUTO" OR ${SPLA_HOST_BLAS} STREQUAL "ARMPL"))
	find_package(ARMPL)
	if(TARGET ARM::pl)
		message(STATUS "Host BLAS Backend: ARMPL")
		list(APPEND SPLA_EXTERNAL_LIBS ARM::pl)
		set(SPLA_BLAS_ARMPL ON)
		set(SPLA_BLAS_HEADER_NAME armpl.h)
		set(_SPLA_BLAS_FOUND TRUE)
	endif()
endif()

if(NOT ${_SPLA_BLAS_FOUND} AND (${SPLA_HOST_BLAS} STREQUAL "AUTO" OR ${SPLA_HOST_BLAS} STREQUAL "BLIS"))
	find_package(BLIS)
	if(TARGET BLIS::blis)
		message(STATUS "Host BLAS Backend: BLIS")
		list(APPEND SPLA_EXTERNAL_LIBS BLIS::blis)
		set(SPLA_BLAS_BLIS ON)
		set(SPLA_BLAS_HEADER_NAME blis.h)
		set(_SPLA_BLAS_FOUND TRUE)
	endif()
endif()

if(NOT ${_SPLA_BLAS_FOUND} AND (${SPLA_HOST_BLAS} STREQUAL "AUTO" OR ${SPLA_HOST_BLAS} STREQUAL "OPENBLAS"))
	find_package(OPENBLAS)
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
	find_package(SCI)
	if(TARGET SCI::sci)
		message(STATUS "Host BLAS Backend: CRAY_LIBSCI")
		list(APPEND SPLA_EXTERNAL_LIBS SCI::sci)
		set(SPLA_BLAS_SCI ON)
		set(SPLA_BLAS_HEADER_NAME cblas.h)
		set(_SPLA_BLAS_FOUND TRUE)
	endif()
endif()

if(NOT ${_SPLA_BLAS_FOUND} AND (${SPLA_HOST_BLAS} STREQUAL "AUTO" OR ${SPLA_HOST_BLAS} STREQUAL "ATLAS"))
	find_package(ATLAS)
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
	find_package(GenericBLAS)
	if(TARGET GenericBLAS::blas)
		message(STATUS "Host BLAS Backend: GENERIC")
		message(STATUS "Host BLAS libs: ${GenericBLAS_LIBRARIES}")
		list(APPEND SPLA_EXTERNAL_LIBS GenericBLAS::blas)
		set(SPLA_BLAS_GENERIC ON)
		set(_SPLA_BLAS_FOUND TRUE)
		# try to find header file
		find_file(_BLAS_HEADER NAMES cblas.h HINTS ${GenericBLAS_INCLUDE_DIRS})
		if(_BLAS_HEADER)
			get_filename_component(SPLA_BLAS_HEADER_NAME ${_BLAS_HEADER} NAME)
		endif()
	endif()
endif()

if(NOT _SPLA_BLAS_FOUND AND NOT ${SPLA_HOST_BLAS} STREQUAL "AUTO")
	message(FATAL_ERROR
		"Could not find selected host blas backend \"${SPLA_HOST_BLAS}\". Set root path or CMAKE_PREFIX_PATH correctly or use \"AUTO\" blas backend for fall back mode.")
endif()

# Fall back to CMake provided FindBLAS as last resort
if(NOT _SPLA_BLAS_FOUND)
	find_package(BLAS REQUIRED)
	message(STATUS "Host BLAS Backend: ${BLAS_LIBRARIES}")
	find_file(_BLAS_HEADER NAMES cblas.h)
	if(_BLAS_HEADER)
		get_filename_component(SPLA_BLAS_HEADER_NAME ${_BLAS_HEADER} NAME)
	endif()
	set(_SPLA_BLAS_FOUND TRUE)
	if(NOT TARGET BLAS::BLAS)
		# target is only available with CMake 3.18.0 and later
		add_library(BLAS::BLAS INTERFACE IMPORTED)
		set_property(TARGET BLAS::BLAS PROPERTY INTERFACE_LINK_LIBRARIES ${BLAS_LIBRARIES} ${BLAS_LINKER_FLAGS})
	endif()
	list(APPEND SPLA_EXTERNAL_LIBS BLAS::BLAS)
	set(SPLA_BLAS_UNKNOWN ON)
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

if(SPLA_BUILD_EXAMPLES)
	add_subdirectory(examples)
endif()

# add tests for developement
if(SPLA_BUILD_TESTS)
	add_subdirectory(tests)
endif()
