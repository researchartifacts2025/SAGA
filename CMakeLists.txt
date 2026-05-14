cmake_minimum_required(VERSION 3.18)
project(saga_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(SAGA_ENABLE_OPENMP "Compile with OpenMP parallelism" ON)
option(SAGA_NATIVE_TUNE   "Apply native-architecture tuning" OFF)

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
