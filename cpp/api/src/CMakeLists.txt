cmake_minimum_required (VERSION 3.9)
enable_language (CUDA)

# Create static library
add_library (cuda STATIC cuda_kernels.cu gpu_vector_ops.cu cpu_vector_ops.cpp)
set_target_properties (cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties (cuda PROPERTIES CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Add source to this project's executable.
add_executable (cuda_test_run main.cpp)
set_property (TARGET cuda_test_run PROPERTY CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Linker
target_link_libraries (cuda_test_run PRIVATE cuda)