cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(spla LANGUAGES CXX VERSION 1.6.1)
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
set(CMAKE_CXX_STANDARD 17)

# Get GNU standard install prefixes
include(GNUInstallDirs)
include(CMakeDependentOption)

#add local module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake/modules)

# Options
option(SPLA_STATIC "Compile as static library" OFF)
option(SPLA_BUILD_TESTS "Build tests" OFF)
option(SPLA_BUILD_EXAMPLES "Compile examples" OFF)
option(SPLA_INSTALL "Enable CMake install commands" ON)
option(SPLA_FORTRAN "Compile fortran module" OFF)
option(SPLA_BUNDLED_TEST_LIBS "Use bundled libraries for building tests" ON)

cmake_dependent_option(SPLA_BUNDLED_GOOGLETEST "Use bundled googletest lib" ON "SPLA_BUNDLED_TEST_LIBS" OFF)
cmake_dependent_option(SPLA_BUNDLED_CLI11 "Use bundled cli11 lib" ON "SPLA_BUNDLED_TEST_LIBS" OFF)

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
    message(FATAL_ERROR "Invalid GPU backend")
  endif()
endif()

if(SPLA_HOST_BLAS)
  message(WARNING "SPLA_HOST_BLAS is no longer in use. Use BLA_VENDOR instead. Check CMake documentation for FindBLAS.")
endif()

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

# CUDA
if(SPLA_CUDA)
  find_package(CUDAToolkit REQUIRED)
  list(APPEND SPLA_EXTERNAL_LIBS CUDA::cudart CUDA::cublas)
endif()

# ROCm
if(SPLA_ROCM)
  find_package(hip CONFIG REQUIRED)
  find_package(rocblas CONFIG REQUIRED)
  list(APPEND SPLA_EXTERNAL_LIBS hip::host roc::rocblas)
endif()


# BLAS
set(BLA_SIZEOF_INTEGER 4)
if(BLA_VENDOR AND "${BLA_VENDOR}" STREQUAL "CRAY_LIBSCI")
  find_package(SCI MODULE REQUIRED)
  list(APPEND SPLA_EXTERNAL_LIBS SCI::sci)
elseif(NOT BLA_VENDOR AND NOT BLAS_LIBRARIES)
  # search in custom order first
  set(_BLAS_VENDOR_LIST Intel10_64lp AOCL_mt Arm_mp OpenBLAS FLAME)
  foreach(BLA_VENDOR IN LISTS _BLAS_VENDOR_LIST)
    find_package(BLAS MODULE QUIET)
    if(BLAS_LIBRARIES)
      message(STATUS "Found BLAS library ${BLA_VENDOR}: ${BLAS_LIBRARIES}")
      break()
    endif()
    message(STATUS "Could NOT find BLAS library ${BLA_VENDOR}")
  endforeach()
  # if not found, search for any BLAS library
  if(NOT BLAS_LIBRARIES)
    unset(BLA_VENDOR)
    find_package(BLAS MODULE REQUIRED)
  endif()
  list(APPEND SPLA_EXTERNAL_LIBS BLAS::BLAS)
else()
  find_package(BLAS MODULE REQUIRED)
  list(APPEND SPLA_EXTERNAL_LIBS BLAS::BLAS)
endif()

if(TARGET BLAS::BLAS)
  # some CMAKE versions (3.18-3.19) don't include libaries in target
  target_link_libraries(BLAS::BLAS INTERFACE ${BLAS_LIBRARIES} ${BLAS_LINKER_FLAGS})
endif()

# check if cblas available
set(CMAKE_REQUIRED_LIBRARIES ${SPLA_EXTERNAL_LIBS})
include(CheckFunctionExists)
unset(SPLA_CBLAS CACHE) # Result is cached, so change of library will not lead to a new check automatically
CHECK_FUNCTION_EXISTS(cblas_zgemm SPLA_CBLAS)

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
