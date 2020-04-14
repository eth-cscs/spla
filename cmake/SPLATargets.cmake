
# Prefer shared library
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/SPLASharedTargets.cmake")
	include("${CMAKE_CURRENT_LIST_DIR}/SPLASharedTargets.cmake")
else()
	include("${CMAKE_CURRENT_LIST_DIR}/SPLAStaticTargets.cmake")
endif()
