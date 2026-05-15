cmake_minimum_required(VERSION 3.18)
project(saga_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(SAGA_ENABLE_OPENMP "Compile with OpenMP parallelism" ON)
option(SAGA_NATIVE_TUNE   "Apply native-architecture tuning" OFF)
option(SAGA_BUILD_CUDA    "Build the CUDA extension (requires nvcc)" OFF)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_native MODULE csrc/saga_native.cpp)
target_compile_features(_native PRIVATE cxx_std_17)

if (SAGA_ENABLE_OPENMP)
  find_package(OpenMP)
  if (OpenMP_CXX_FOUND)
    target_link_libraries(_native PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "saga_native: OpenMP enabled")
  else()
    message(WARNING "saga_native: OpenMP requested but not found; falling back to single-threaded")
  endif()
endif()

if (SAGA_NATIVE_TUNE)
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(_native PRIVATE -O3 -march=native)
  elseif (MSVC)
    target_compile_options(_native PRIVATE /O2)
  endif()
endif()

if (NOT CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set_target_properties(_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/saga"
  )
endif()

install(TARGETS _native LIBRARY DESTINATION saga)

# ----------------------------------------------------------------- CUDA
#
# CUDA build is opt-in because it requires nvcc and a torch install. Toggle
# with -DSAGA_BUILD_CUDA=ON. The CUDA kernels themselves live in
# csrc/cuda/; see setup_cuda.py for the recommended pybind11+torch build
# path. This CMake target exists for HPC sites that prefer a single
# canonical build system.

if (SAGA_BUILD_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_ARCHITECTURES 70 80 90)

  find_package(Torch CONFIG REQUIRED)

  pybind11_add_module(_cuda MODULE
    csrc/cuda/prefetch_stream.cu
    csrc/cuda/migration.cu
    csrc/cuda/prefix_overlap.cu
    csrc/cuda/walru_score_cuda.cu
    csrc/cuda/compact_pool.cu
    csrc/cuda/paged_attention_walru.cu
    csrc/cuda/saga_cuda_pybind.cpp
  )
  target_include_directories(_cuda PRIVATE csrc/cuda)
  target_link_libraries(_cuda PRIVATE ${TORCH_LIBRARIES})
  target_compile_options(_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math --expt-relaxed-constexpr --extended-lambda>
  )

  set_target_properties(_cuda PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/saga"
  )
  install(TARGETS _cuda LIBRARY DESTINATION saga)
endif()
