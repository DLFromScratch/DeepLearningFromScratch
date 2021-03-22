# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.9)
project ("cpp" LANGUAGES CXX)

# Include sub-projects.
add_subdirectory ("api")
#add_subdirectory ("train")