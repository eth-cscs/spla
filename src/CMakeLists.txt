set(SPLA_SOURCE_FILES
	block_generation/block_cyclic_generator.cpp
	util/blas_interface.cpp
	timing/rt_graph.cpp
	pgemm_ssb/pgemm_ssb_host.cpp
	pgemm_ssb/ring_reduce_tile_host.cpp
	pgemm_ssb/add_kernel.cpp
	pgemm_sbs/pgemm_sbs_host.cpp
	pgemm_sbs/stripe_host.cpp
	gemm/gemm_host.cpp
	spla/matrix_distribution.cpp
	spla/matrix_distribution_internal.cpp
	spla/pgemm_ssb.cpp
	spla/pgemm_sbs.cpp
	spla/context.cpp
	spla/gemm.cpp
	)

if(SPLA_CUDA OR SPLA_ROCM)
	list(APPEND SPLA_SOURCE_FILES
				pgemm_ssb/ring_reduce_tile_gpu.cpp
				pgemm_ssb/pgemm_ssb_gpu.cpp
				pgemm_sbs/stripe_gpu.cpp
				pgemm_sbs/pgemm_sbs_gpu.cpp
				gemm/gemm_gpu.cpp
				gpu_util/multiply_gpu.cpp
		)
endif()

if(SPLA_TIMING)
	list(APPEND SPLA_SOURCE_FILES timing/timing.cpp)
endif()

add_library(spla ${SPLA_LIBRARY_TYPE}
	${SPLA_SOURCE_FILES}
	)

# only enforces minimum standard
# target_compile_features(spla PUBLIC cxx_std_11)

# Don't export any symbols of external static libaries. Only works on linux.
if(UNIX AND NOT APPLE)
	if(${CMAKE_VERSION} VERSION_LESS "3.13.5") 
		set(CMAKE_SHARED_LINKER_FLAGS ${CMAKE_SHARED_LINKER_FLAGS} "-Wl,--exclude-libs,ALL")
	else()
		target_link_options(spla PRIVATE "-Wl,--exclude-libs,ALL")
	endif()
endif()

# generate export header to control symbol visibility
include(GenerateExportHeader)
generate_export_header(spla)
configure_file("${CMAKE_CURRENT_BINARY_DIR}/spla_export.h"
	"${PROJECT_BINARY_DIR}/spla/spla_export.h"
	COPYONLY
)

target_compile_options(spla PRIVATE ${SPLA_DEFINITIONS} ${SPLA_EXTERNAL_COMPILE_OPTIONS})
target_include_directories(spla PRIVATE ${SPLA_EXTERNAL_INCLUDE_DIRS})
target_include_directories(spla PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(spla PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(spla PRIVATE ${PROJECT_BINARY_DIR})
target_link_libraries(spla PUBLIC ${SPLA_EXTERNAL_LIBS})

target_include_directories(spla INTERFACE ${SPLA_INTERFACE_INCLUDE_DIRS})
target_include_directories(spla INTERFACE $<INSTALL_INTERFACE:include>) # for install(EXPORT ...)
target_include_directories(spla INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>) # for export(...)

# set list of interface libries for exporting (workaround for bug with exporting static library:
# cmake uses MPI_CXX and OpenMP::OpenMP_CXX targets, which are not known externally)
if(POLICY CMP0022)
	set_target_properties(spla PROPERTIES INTERFACE_LINK_LIBRARIES "${SPLA_INTERFACE_LIBS}")
else()
	set_target_properties(spla PROPERTIES LINK_INTERFACE_LIBRARIES "${SPLA_INTERFACE_LIBS}")
endif()

# set packge config names
if(SPLA_STATIC)
	set(SPLA_VERSION_FILE "SPLAStaticConfigVersion.cmake")
	set(SPLA_CONFIG_FILE "SPLAStaticConfig.cmake")
	set(SPLA_TARGETS_FILE "SPLAStaticTargets.cmake")
else()
	set(SPLA_VERSION_FILE "SPLASharedConfigVersion.cmake")
	set(SPLA_CONFIG_FILE "SPLASharedConfig.cmake")
	set(SPLA_TARGETS_FILE "SPLASharedTargets.cmake")
endif()


# generate cmake package
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
	"${PROJECT_BINARY_DIR}/${SPLA_VERSION_FILE}"
	VERSION ${Upstream_VERSION}
	COMPATIBILITY AnyNewerVersion
)
export(TARGETS spla NAMESPACE SPLA:: FILE ${PROJECT_BINARY_DIR}/${SPLA_TARGETS_FILE})
configure_file(${PROJECT_SOURCE_DIR}/cmake/${SPLA_CONFIG_FILE}
	"${PROJECT_BINARY_DIR}/${SPLA_CONFIG_FILE}"
	@ONLY
)
configure_file(${PROJECT_SOURCE_DIR}/cmake/SPLAConfig.cmake
	"${PROJECT_BINARY_DIR}/SPLAConfig.cmake"
	COPYONLY
)
configure_file(${PROJECT_SOURCE_DIR}/cmake/SPLAConfigVersion.cmake
	"${PROJECT_BINARY_DIR}/SPLAConfigVersion.cmake"
	COPYONLY
)
configure_file(${PROJECT_SOURCE_DIR}/cmake/SPLATargets.cmake
	"${PROJECT_BINARY_DIR}/SPLATargets.cmake"
	COPYONLY
)
configure_file(${PROJECT_SOURCE_DIR}/cmake/SPLA.pc.in
	"${PROJECT_BINARY_DIR}/SPLA.pc"
	@ONLY
)

# installation commands
if(SPLA_INSTALL)
	install(TARGETS spla DESTINATION ${CMAKE_INSTALL_LIBDIR} EXPORT SPLATargets)
	install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/spla DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.f90")
	install(FILES ${PROJECT_BINARY_DIR}/spla/config.h "${PROJECT_BINARY_DIR}/spla/spla_export.h" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/spla)
	install(EXPORT SPLATargets NAMESPACE SPLA:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SPLA FILE ${SPLA_TARGETS_FILE})
	install(
	  FILES
		"${PROJECT_BINARY_DIR}/SPLAConfig.cmake"
		"${PROJECT_BINARY_DIR}/SPLATargets.cmake"
		"${PROJECT_BINARY_DIR}/SPLAConfigVersion.cmake"
		"${PROJECT_BINARY_DIR}/${SPLA_CONFIG_FILE}"
		"${PROJECT_BINARY_DIR}/${SPLA_VERSION_FILE}"
	  DESTINATION
	    ${CMAKE_INSTALL_LIBDIR}/cmake/SPLA
	)

	install(FILES ${PROJECT_BINARY_DIR}/SPLA.pc DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)

	if(SPLA_FORTRAN)
		install(FILES ${PROJECT_BINARY_DIR}/src/spla.mod DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/spla)
	endif()
endif()
