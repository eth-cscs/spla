set(SPLA_SOURCE_FILES
	block_generation/block_cyclic_generator.cpp
	util/blas_interface.cpp
	timing/rt_graph.cpp
	timing/timing.cpp
	pgemm_ssb/pgemm_ssb_host.cpp
	pgemm_ssb/ring_ssb_host.cpp
	pgemm_ssb/add_kernel.cpp
	pgemm_sbs/pgemm_sbs_host.cpp
	pgemm_sbs/ring_sbs_host.cpp
	gemm/gemm_host.cpp
	spla/matrix_distribution.cpp
	spla/matrix_distribution_internal.cpp
	spla/pgemm_ssb.cpp
	spla/pgemm_ssbtr.cpp
	spla/pgemm_sbs.cpp
	spla/context.cpp
	spla/gemm.cpp
	)

if(SPLA_CUDA OR SPLA_ROCM)
	list(APPEND SPLA_SOURCE_FILES
				pgemm_ssb/ring_ssb_gpu.cpp
				pgemm_ssb/pgemm_ssb_gpu.cpp
				pgemm_sbs/pgemm_sbs_gpu.cpp
            	pgemm_sbs/ring_sbs_gpu.cpp
				gemm/gemm_gpu.cpp
				gpu_util/multiply_gpu.cpp
		)
endif()

# Creates spla library with given name. All common target modifications should be done here.
macro(spla_create_library _TARGET_NAME)
	add_library(${_TARGET_NAME} ${SPLA_LIBRARY_TYPE}
		${SPLA_SOURCE_FILES}
		)

	set_property(TARGET ${_TARGET_NAME} PROPERTY VERSION ${SPLA_VERSION})
	set_property(TARGET ${_TARGET_NAME} PROPERTY SOVERSION ${SPLA_SO_VERSION})


	# Don't export any symbols of external static libaries. Only works on linux.
	if(UNIX AND NOT APPLE)
		if(${CMAKE_VERSION} VERSION_LESS "3.13.5") 
			set(CMAKE_SHARED_LINKER_FLAGS ${CMAKE_SHARED_LINKER_FLAGS} "-Wl,--exclude-libs,ALL")
		else()
			target_link_options(${_TARGET_NAME} PRIVATE "-Wl,--exclude-libs,ALL")
		endif()
	endif()


	target_compile_options(${_TARGET_NAME} PRIVATE ${SPLA_DEFINITIONS} ${SPLA_EXTERNAL_COMPILE_OPTIONS})
	target_include_directories(${_TARGET_NAME} PRIVATE ${SPLA_EXTERNAL_INCLUDE_DIRS})
	target_include_directories(${_TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include)
	target_include_directories(${_TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/src)
	target_include_directories(${_TARGET_NAME} PRIVATE ${PROJECT_BINARY_DIR})


	target_link_libraries(${_TARGET_NAME} PRIVATE ${SPLA_EXTERNAL_LIBS})
	# SPLA inteface needs MPI to be compiled. Avoid requiring CXX language for installed cmake config files, so only make public for build.
	target_link_libraries(${_TARGET_NAME} PUBLIC $<BUILD_INTERFACE:MPI::MPI_CXX>)

	target_include_directories(${_TARGET_NAME} INTERFACE ${SPLA_INTERFACE_INCLUDE_DIRS})
	target_include_directories(${_TARGET_NAME} INTERFACE $<INSTALL_INTERFACE:include>) # for install(EXPORT ...)
	target_include_directories(${_TARGET_NAME} INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>) # for export(...)
	if(${SPLA_FORTRAN}) # Add include directory for fortran module
		target_include_directories(${_TARGET_NAME} INTERFACE $<INSTALL_INTERFACE:include/spla>)
		target_include_directories(${_TARGET_NAME} INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>)
	endif()

endmacro()

spla_create_library(spla)
set_target_properties(spla PROPERTIES VISIBILITY_INLINES_HIDDEN TRUE CXX_VISIBILITY_PRESET hidden)

if(SPLA_BUILD_TESTS)
	# create library with default visibility if tests are build, to allow linking to internal symbols
	spla_create_library(spla_test)
	set_target_properties(spla_test PROPERTIES VISIBILITY_INLINES_HIDDEN FALSE CXX_VISIBILITY_PRESET default)
	target_compile_options(spla_test PUBLIC -DSPLA_STATIC_DEFINE)
	# enable internal timings
	target_compile_options(spla_test PUBLIC -DSPLA_TIMING)
endif()

# generate export header to control symbol visibility
include(GenerateExportHeader)
generate_export_header(spla)
configure_file("${CMAKE_CURRENT_BINARY_DIR}/spla_export.h"
	"${PROJECT_BINARY_DIR}/spla/spla_export.h"
	COPYONLY
)

# build fortran module
if(SPLA_FORTRAN)
	add_library(spla_fortran OBJECT ${PROJECT_SOURCE_DIR}/include/spla/spla.f90)
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

	install(DIRECTORY "${PROJECT_SOURCE_DIR}/cmake/modules"
		DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/SPLA"
        FILES_MATCHING PATTERN "*.cmake")

	if(SPLA_FORTRAN)
		install(FILES ${PROJECT_BINARY_DIR}/src/spla.mod DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/spla)
	endif()
endif()
