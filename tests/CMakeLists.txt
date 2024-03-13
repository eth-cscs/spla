cmake_minimum_required(VERSION 3.14 FATAL_ERROR) # FetchContent_MakeAvailable requires at least 3.14
set(SPLA_TEST_LIBRARIES)

# update time stamps when using FetchContent
if(POLICY CMP0135)
	cmake_policy(SET CMP0135 NEW)
endif()

set(BUILD_GMOCK OFF CACHE BOOL "")
set(INSTALL_GTEST OFF CACHE BOOL "")
mark_as_advanced(BUILD_GMOCK INSTALL_GTEST)
include(FetchContent)

# add googletest
if(SPLA_BUNDLED_GOOGLETEST)
	FetchContent_Declare(
		googletest
		URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.tar.gz
		URL_MD5 95b29f0038ec84a611df951d74d99897
	)
	FetchContent_MakeAvailable(googletest)
else()
  find_package(googletest CONFIG REQUIRED)
endif()
list(APPEND SPLA_TEST_LIBRARIES gtest_main)

# add gtest_mpi
if(SPLA_BUNDLED_GTEST_MPI)
	FetchContent_Declare(
		gtest_mpi
		URL https://github.com/AdhocMan/gtest_mpi/archive/refs/tags/v1.1.0.tar.gz
		URL_MD5 0fea4d6a8a8dce9a597d5dceab15e3d1
	)
	FetchContent_MakeAvailable(gtest_mpi)
else()
  find_package(gtest_mpi CONFIG REQUIRED)
endif()
list(APPEND SPLA_TEST_LIBRARIES gtest_mpi)

# add command line parser
if(SPLA_BUNDLED_GTEST_MPI)
	FetchContent_Declare(
		cli11
		URL https://github.com/CLIUtils/CLI11/archive/refs/tags/v2.3.2.tar.gz
		URL_MD5 b80cb645dee25982110b068b426363ff
	)
	FetchContent_MakeAvailable(cli11)
else()
	find_package(CLI11 CONFIG REQUIRED)
endif()
list(APPEND SPLA_TEST_LIBRARIES CLI11::CLI11)


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
elseif(SPLA_BLAS_SCI)
	list(APPEND SPLA_TEST_LIBRARIES SCI::sci_mpi)
else()
	find_package(SCALAPACK REQUIRED)
	list(APPEND SPLA_TEST_LIBRARIES SCALAPACK::SCALAPACK)
endif()

if(UNIX AND NOT APPLE)
	# on Daint, dl library appears to be required
	find_library(SPLA_DL_LIBRARY dl)
	if(SPLA_DL_LIBRARY)
		list(APPEND SPLA_TEST_LIBRARIES ${SPLA_DL_LIBRARY})
	endif()
endif()

add_executable(run_tests programs/run_tests.cpp test_pool_allocator.cpp test_gemm.cpp test_gemm_ssb.cpp test_gemm_sbs.cpp)
target_link_libraries(run_tests PUBLIC spla_test ${SPLA_EXTERNAL_LIBS} ${SPLA_TEST_LIBRARIES})
target_include_directories(run_tests PUBLIC ${SPLA_INCLUDE_DIRS} ${SPLA_EXTERNAL_INCLUDE_DIRS})
target_compile_options(run_tests PRIVATE ${SPLA_DEFINITIONS} ${SPLA_EXTERNAL_COMPILE_OPTIONS})

add_executable(benchmark programs/benchmark.cpp)
target_link_libraries(benchmark PUBLIC spla_test ${SPLA_EXTERNAL_LIBS} ${SPLA_TEST_LIBRARIES})
target_include_directories(benchmark PUBLIC ${SPLA_INCLUDE_DIRS} ${SPLA_EXTERNAL_INCLUDE_DIRS})
target_compile_options(benchmark PRIVATE ${SPLA_DEFINITIONS} ${SPLA_EXTERNAL_COMPILE_OPTIONS})

add_executable(benchmark_scalapack programs/benchmark_scalapack.cpp)
target_link_libraries(benchmark_scalapack PUBLIC spla_test ${SPLA_EXTERNAL_LIBS} ${SPLA_TEST_LIBRARIES})
target_include_directories(benchmark_scalapack PUBLIC ${SPLA_INCLUDE_DIRS} ${SPLA_EXTERNAL_INCLUDE_DIRS})
target_compile_options(benchmark_scalapack PRIVATE ${SPLA_DEFINITIONS} ${SPLA_EXTERNAL_COMPILE_OPTIONS})
