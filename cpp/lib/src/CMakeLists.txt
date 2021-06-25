# CMakeList.txt : CMake project for api, include source and define
# project specific logic here.
#

add_executable (xxx "xxx.cpp")
add_library (xxx "xxx.cpp" "xxx.cpp" "xxx.cpp" "xxx.cpp")

# Add libraries
target_link_libraries (xxx xxx)

# Add CUDA ops if necessary
if (BUILD_CUDA)
	add_subdirectory("cuda_ops")
endif()