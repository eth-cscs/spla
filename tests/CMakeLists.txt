cmake_minimum_required(VERSION 3.11 FATAL_ERROR) # git fetch module requires at least 3.11
set(SPLA_TEST_LIBRARIES)
set(SPLA_TEST_INCLUDE_DIRS)

set(BUILD_GMOCK OFF CACHE BOOL "")
set(INSTALL_GTEST OFF CACHE BOOL "")
mark_as_advanced(BUILD_GMOCK INSTALL_GTEST)
include(FetchContent)

# add googletest
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.8.1
)
FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
  message(STATUS "Downloading Google Test repository...")
  FetchContent_Populate(googletest)
endif()
add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
list(APPEND SPLA_TEST_LIBRARIES gtest_main)

# add gtest_mpi
FetchContent_Declare(
  gtest_mpi
  GIT_REPOSITORY https://github.com/AdhocMan/gtest_mpi.git
  GIT_TAG        v1.0.0
)
FetchContent_GetProperties(gtest_mpi)
if(NOT gtest_mpi_POPULATED)
  message(STATUS "Downloading Google Test MPI extension repository...")
  FetchContent_Populate(gtest_mpi)
endif()
add_subdirectory(${gtest_mpi_SOURCE_DIR} ${gtest_mpi_BINARY_DIR})
list(APPEND SPLA_TEST_LIBRARIES gtest_mpi)

# add command line parser
FetchContent_Declare(
  cli11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG        v1.7.1
)
FetchContent_GetProperties(cli11)
if(NOT cli11_POPULATED)
  message(STATUS "Downloading CLI11 command line parser repository...")
  FetchContent_Populate(cli11)
endif()
list(APPEND SPLA_TEST_INCLUDE_DIRS ${cli11_SOURCE_DIR}/include)


if(SPLA_BLAS_MKL)
	set(_MKL_MPI mpich)
	if(MPIEXEC_EXECUTABLE)
		execute_process(COMMAND ${MPIEXEC_EXECUTABLE} --version OUTPUT_VARIABLE _MPIRUN_OUTPUT)
		string(FIND "${_MPIRUN_OUTPUT}" "Open MPI" _OMPI_POS)
		if(NOT _OMPI_POS STREQUAL "-1")
			set(_MKL_MPI ompi)
		endif()
	endif()
	list(APPEND SPLA_TEST_LIBRARIES mkl::scalapack_${_MKL_MPI}_${SPLA_MKL_TARGET})
else()
	find_package(SCALAPACK REQUIRED)
	list(APPEND SPLA_TEST_LIBRARIES SCALAPACK::SCALAPACK)
endif()

if(UNIX AND NOT APPLE)
	# on Daint, dl library appears to be required
	find_library(SPLA_DL_LIBRARY dl)
	list(APPEND SPLA_TEST_LIBRARIES ${SPLA_DL_LIBRARY})
endif()

add_executable(run_tests programs/run_tests.cpp test_pool_allocator.cpp test_gemm.cpp test_gemm_ssb.cpp test_gemm_sbs.cpp)
target_link_libraries(run_tests PUBLIC spla_test ${SPLA_EXTERNAL_LIBS} ${SPLA_TEST_LIBRARIES})
target_include_directories(run_tests PUBLIC ${SPLA_INCLUDE_DIRS} ${SPLA_EXTERNAL_INCLUDE_DIRS} ${SPLA_TEST_INCLUDE_DIRS})
target_compile_options(run_tests PRIVATE ${SPLA_DEFINITIONS} ${SPLA_EXTERNAL_COMPILE_OPTIONS})

add_executable(benchmark programs/benchmark.cpp)
target_link_libraries(benchmark PUBLIC spla_test ${SPLA_EXTERNAL_LIBS})
target_include_directories(benchmark PUBLIC ${SPLA_INCLUDE_DIRS} ${SPLA_EXTERNAL_INCLUDE_DIRS} ${SPLA_TEST_INCLUDE_DIRS})
target_compile_options(benchmark PRIVATE ${SPLA_DEFINITIONS} ${SPLA_EXTERNAL_COMPILE_OPTIONS})

add_executable(benchmark_scalapack programs/benchmark_scalapack.cpp)
target_link_libraries(benchmark_scalapack PUBLIC spla_test ${SPLA_EXTERNAL_LIBS} ${SPLA_TEST_LIBRARIES})
target_include_directories(benchmark_scalapack PUBLIC ${SPLA_INCLUDE_DIRS} ${SPLA_EXTERNAL_INCLUDE_DIRS} ${SPLA_TEST_INCLUDE_DIRS})
target_compile_options(benchmark_scalapack PRIVATE ${SPLA_DEFINITIONS} ${SPLA_EXTERNAL_COMPILE_OPTIONS})
