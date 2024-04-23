include(CMakeFindDependencyMacro)
macro(find_dependency_components)
    if(${ARGV0}_FOUND AND ${CMAKE_VERSION} VERSION_LESS "3.15.0")
        # find_dependency does not handle new components correctly before 3.15.0
        set(${ARGV0}_FOUND FALSE)
    endif()
    find_dependency(${ARGV})
endmacro()

# Only look for modules we installed and save value
set(_CMAKE_MODULE_PATH_SAVE ${CMAKE_MODULE_PATH})
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/modules")

# options used for building library
set(SPLA_STATIC @SPLA_STATIC@)
set(SPLA_GPU_BACKEND @SPLA_GPU_BACKEND@)
set(SPLA_BUILD_TESTS @SPLA_BUILD_TESTS@)
set(SPLA_TIMING @SPLA_TIMING@)
set(SPLA_FORTRAN @SPLA_FORTRAN@)

# make sure CXX is enabled
get_property(_LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
if(SPLA_FIND_REQUIRED AND NOT "CXX" IN_LIST _LANGUAGES)
    message(FATAL_ERROR "SPLA requires CXX language to be enabled for static linking.")
endif()

# find required targets
if(NOT TARGET MPI::MPI_CXX)
    find_dependency_components(MPI COMPONENTS CXX)
endif()

if("C" IN_LIST _LANGUAGES AND NOT TARGET MPI::MPI_C)
    find_dependency_components(MPI COMPONENTS C)
endif()

if("Fortran" IN_LIST _LANGUAGES AND NOT TARGET MPI::MPI_Fortran)
    find_dependency_components(MPI COMPONENTS Fortran)
endif()

if(SPLA_OMP)
    if(NOT TARGET OpenMP::OpenMP_CXX)
        find_dependency_components(OpenMP COMPONENTS CXX)
    endif()
endif()

if(SPLA_ROCM)
    find_dependency(hip CONFIG)
    find_dependency(rocblas CONFIG)
endif()


if(SPLA_CUDA)
    if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.17.0") 
        find_dependency(CUDAToolkit)
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
endif()

find_dependency(BLASExt MODULE)

set(CMAKE_MODULE_PATH ${_CMAKE_MODULE_PATH_SAVE}) # restore module path

# find_dependency may set SPLA_FOUND to false, so only add spla if everything required was found
if(NOT DEFINED SPLA_FOUND OR SPLA_FOUND)
    # add version of package
    include("${CMAKE_CURRENT_LIST_DIR}/SPLAStaticConfigVersion.cmake")

    # add library target
    include("${CMAKE_CURRENT_LIST_DIR}/SPLAStaticTargets.cmake")

    target_link_libraries(SPLA::spla INTERFACE MPI::MPI_CXX)
    if(TARGET MPI::MPI_C)
        target_link_libraries(SPLA::spla INTERFACE MPI::MPI_C)
    endif()
    if(TARGET MPI::MPI_Fortran)
        target_link_libraries(SPLA::spla INTERFACE MPI::MPI_Fortran)
    endif()
endif()
