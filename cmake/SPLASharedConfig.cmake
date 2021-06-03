include(CMakeFindDependencyMacro)

# options used for building library
set(SPLA_OMP @SPLA_OMP@)
set(SPLA_STATIC @SPLA_STATIC@)
set(SPLA_GPU_BACKEND @SPLA_GPU_BACKEND@)
set(SPLA_BUILD_TESTS @SPLA_BUILD_TESTS@)
set(SPLA_TIMING @SPLA_TIMING@)
set(SPLA_FORTRAN @SPLA_FORTRAN@)

# SPLA only has MPI as public dependency, since the mpi header is
# part of the public header file.
# Only look for MPI if header matching language is possibly used
get_property(_LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
if("CXX" IN_LIST _LANGUAGES)
	find_dependency(MPI COMPONENTS CXX)
endif()

if("C" IN_LIST _LANGUAGES)
	find_dependency(MPI COMPONENTS C)
endif()

if("Fortran" IN_LIST _LANGUAGES)
	find_dependency(MPI COMPONENTS Fortran)
endif()

# find_dependency may set SPLA_FOUND to false, so only add spla if everything required was found
if(NOT DEFINED SPLA_FOUND OR SPLA_FOUND)
	# add version of package
	include("${CMAKE_CURRENT_LIST_DIR}/SPLASharedConfigVersion.cmake")

	# add library target
	include("${CMAKE_CURRENT_LIST_DIR}/SPLASharedTargets.cmake")

	target_link_libraries(SPLA::spla INTERFACE MPI::MPI_CXX)
	if("C" IN_LIST _LANGUAGES)
		target_link_libraries(SPLA::spla INTERFACE MPI::MPI_C)
	endif()
	if("Fortran" IN_LIST _LANGUAGES)
		target_link_libraries(SPLA::spla INTERFACE MPI::MPI_Fortran)
	endif()
endif()
