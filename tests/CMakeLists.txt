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
  find_package(GTest CONFIG REQUIRED)
endif()
list(APPEND SPLA_TEST_LIBRARIES GTest::gtest)

# add command line parser
if(SPLA_BUNDLED_CLI11)
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

find_package(SCALAPACK MODULE REQUIRED)
list(APPEND SPLA_TEST_LIBRARIES SCALAPACK::SCALAPACK)

if(UNIX AND NOT APPLE)
	# on Daint, dl library appears to be required
	find_library(SPLA_DL_LIBRARY dl)
	if(SPLA_DL_LIBRARY)
		list(APPEND SPLA_TEST_LIBRARIES ${SPLA_DL_LIBRARY})
	endif()
endif()

add_executable(run_tests programs/run_tests.cpp test_pool_allocator.cpp test_gemm.cpp test_gemm_ssb.cpp test_gemm_sbs.cpp gtest_mpi.cpp)
target_link_libraries(run_tests PUBLIC spla_test ${SPLA_EXTERNAL_LIBS} ${SPLA_TEST_LIBRARIES})
target_include_directories(run_tests PUBLIC ${SPLA_INCLUDE_DIRS} ${SPLA_EXTERNAL_INCLUDE_DIRS} ${CMAKE_CURRENT_LIST_DIR})
target_compile_options(run_tests PRIVATE ${SPLA_DEFINITIONS} ${SPLA_EXTERNAL_COMPILE_OPTIONS})

add_executable(benchmark programs/benchmark.cpp)
target_link_libraries(benchmark PUBLIC spla_test ${SPLA_EXTERNAL_LIBS} ${SPLA_TEST_LIBRARIES})
target_include_directories(benchmark PUBLIC ${SPLA_INCLUDE_DIRS} ${SPLA_EXTERNAL_INCLUDE_DIRS})
target_compile_options(benchmark PRIVATE ${SPLA_DEFINITIONS} ${SPLA_EXTERNAL_COMPILE_OPTIONS})

add_executable(benchmark_scalapack programs/benchmark_scalapack.cpp)
target_link_libraries(benchmark_scalapack PUBLIC spla_test ${SPLA_EXTERNAL_LIBS} ${SPLA_TEST_LIBRARIES})
target_include_directories(benchmark_scalapack PUBLIC ${SPLA_INCLUDE_DIRS} ${SPLA_EXTERNAL_INCLUDE_DIRS})
target_compile_options(benchmark_scalapack PRIVATE ${SPLA_DEFINITIONS} ${SPLA_EXTERNAL_COMPILE_OPTIONS})
