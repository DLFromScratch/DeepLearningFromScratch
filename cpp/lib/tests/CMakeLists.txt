# Unit test CMake file

# Include and library directories
set (GTEST_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/lib/third-party/gtest/include")
set (GTEST_LIB_DIR "${CMAKE_SOURCE_DIR}/lib/third-party/gtest/lib")
include_directories (${GTEST_INCLUDE_DIR})

# Add executable for unit tests
add_executable (tests "xxx.cpp" "xxx.cpp")

# Add libraries for tests
find_library (GTEST NAMES gtest PATHS ${GTEST_LIB_DIR})
target_link_libraries (tests utils ${GTEST})

# Add GoogleTest so that Visual Studio can add tests to Test Explorer
include (GoogleTest)
gtest_discover_tests (tests)