# CMakeList.txt : CMake project for api, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.9)
project ("api" LANGUAGES CXX)

# Include sub-directory
add_subdirectory("src")

# Add testing if necessary
if (BUILD_TESTING)
	enable_testing ()
	add_subdirectory ("tests")
endif()