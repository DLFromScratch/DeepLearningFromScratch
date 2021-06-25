# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.9)
project ("cpp" LANGUAGES CXX)

# Build options
option (BUILD_SHARED_LIBS "Build shared libraries" OFF)
option (BUILD_TESTING "Build unit tests" OFF)

# Include sub-projects.
add_subdirectory ("lib")
#add_subdirectory ("train")