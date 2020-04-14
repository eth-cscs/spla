# options used for building library
set(SPLA_OMP @SPLA_OMP@)
set(SPLA_GPU_BACKEND @SPLA_GPU_BACKEND@)

# add version of package
include("${CMAKE_CURRENT_LIST_DIR}/SPLASharedConfigVersion.cmake")

# add library target
include("${CMAKE_CURRENT_LIST_DIR}/SPLASharedTargets.cmake")

