cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(RKCKSolverCuda LANGUAGES CXX CUDA)

include_directories("$ENV{CUDA_PATH}\\include\\")
include_directories(".\\")

add_executable(Driver "driver.cu" "solver.cu" "solver.h" "system_params.h")

target_compile_features(Driver PUBLIC cxx_std_11)