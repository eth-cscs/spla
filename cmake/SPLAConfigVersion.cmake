
# Prefer shared library
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/SPLASharedConfigVersion.cmake")
	include("${CMAKE_CURRENT_LIST_DIR}/SPLASharedConfigVersion.cmake")
else()
	include("${CMAKE_CURRENT_LIST_DIR}/SPLAStaticConfigVersion.cmake")
endif()
