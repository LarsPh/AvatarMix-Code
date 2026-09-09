# Detects whether this is a top-level project
get_directory_property(HAS_PARENT PARENT_DIRECTORY)
if(HAS_PARENT)
    set(POLYFEM_TOPLEVEL_PROJECT OFF)
else()
    set(POLYFEM_TOPLEVEL_PROJECT ON)
endif()

# Check required CMake version
set(REQUIRED_CMAKE_VERSION "3.25.0")
if(POLYFEM_TOPLEVEL_PROJECT)
    cmake_minimum_required(VERSION ${REQUIRED_CMAKE_VERSION})
else()
    # Don't use cmake_minimum_required here to avoid implicitly overriding parent policies
    if(${CMAKE_VERSION} VERSION_LESS ${REQUIRED_CMAKE_VERSION})
        message(FATAL_ERROR "CMake required version to build PolyFEM is ${REQUIRED_CMAKE_VERSION}")
    endif()
endif()

# Include user-provided default options if available. We do that before the main
# `project()` so that we can define the C/C++ compilers from the option file.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/PolyFEMOptions.cmake)
    message(STATUS "Using local options file: ${CMAKE_CURRENT_SOURCE_DIR}/PolyFEMOptions.cmake")
    include(${CMAKE_CURRENT_SOURCE_DIR}/PolyFEMOptions.cmake)
endif()

# Enable ccache if available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  option(POLYFEM_WITH_CCACHE "Enable ccache when building PolyFEM" ${POLYFEM_TOPLEVEL_PROJECT})
else()
  option(POLYFEM_WITH_CCACHE "Enable ccache when building PolyFEM" OFF)
endif()
if(POLYFEM_WITH_CCACHE AND CCACHE_PROGRAM)
  message(STATUS "Enabling Ccache support")
  set(ccacheEnv
    CCACHE_BASEDIR=${CMAKE_BINARY_DIR}
    CCACHE_SLOPPINESS=clang_index_store,include_file_ctime,include_file_mtime,locale,pch_defines,time_macros
  )
  foreach(lang IN ITEMS C CXX)
    set(CMAKE_${lang}_COMPILER_LAUNCHER
      ${CMAKE_COMMAND} -E env ${ccacheEnv} ${CCACHE_PROGRAM}
    )
  endforeach()
endif()

################################################################################
# CMake Policies
################################################################################

cmake_policy(SET CMP0054 NEW) # Only interpret if() arguments as variables or keywords when unquoted.
cmake_policy(SET CMP0076 NEW) # target_sources() command converts relative paths to absolute.
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
  cmake_policy(SET CMP0135 NEW) # Set the timestamps of all extracted contents to the time of the extraction.
endif()

################################################################################
project(PolyFEM DESCRIPTION "A polyvalent C++ FEM library" LANGUAGES C CXX)

# Polyfem options
set(POLYFEM_SMALL_N  80 CACHE STRING "Maximum length for stack-allocated vectors (gradient + Hessian).")
set(POLYFEM_BIG_N   1000 CACHE STRING "Maximum length for stack-allocated vectors (gradient only).")

# Polyfem options for enabling/disabling optional libraries
option(POLYFEM_WITH_TESTS     "Build tests"                                 ON)
set(POLYFEM_THREADING "TBB" CACHE STRING "Multithreading library to use (options: CPP, TBB, NONE)")
set_property(CACHE POLYFEM_THREADING PROPERTY STRINGS "CPP" "TBB" "NONE")
option(POLYFEM_CODE_COVERAGE "Enable coverage reporting" OFF)
option(POLYFEM_WITH_PARAVIEWO "Enable advanced export with paraviewo" OFF)
option(POLYFEM_WITH_TENSORBOARD "Enable TensorBoard scalar logging" OFF)

set(POLYSOLVE_WITH_AMGCL OFF)

add_library(polyfem_coverage_config INTERFACE)
if(POLYFEM_CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # Add required flags (GCC & LLVM/Clang)
  target_compile_options(polyfem_coverage_config INTERFACE
    -g         # generate debug info
    --coverage # sets all required flags
  )
  target_link_options(polyfem_coverage_config INTERFACE --coverage)
endif()

# Set default minimum C++ standard
if(POLYFEM_TOPLEVEL_PROJECT)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

### Configuration
set(POLYFEM_SOURCE_DIR "${PROJECT_SOURCE_DIR}/src/polyfem")
set(POLYFEM_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/src")

# set(CPM_polysolve_SOURCE "/Users/zizhouhuang/Desktop/polysolve")
# set(CPM_ipc-toolkit_SOURCE "/Users/zizhouhuang/Desktop/ipc-toolkit")

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/polyfem/")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/recipes/")

# General CMake utils
include(polyfem_autogen)
include(polyfem_cpm_cache)
include(polyfem_use_colors)

# Generate position independent code by default
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

################################################################################
# Polyfem library
################################################################################

# Add an empty library and fill in the list of sources in `src/CMakeLists.txt`.
add_library(polyfem)
add_library(polyfem::polyfem ALIAS polyfem)

target_link_libraries(polyfem PUBLIC polyfem_coverage_config)

# Add source and header files to polyfem
add_subdirectory("${POLYFEM_SOURCE_DIR}")

# Public include directory for PolyFEM
target_include_directories(polyfem PUBLIC "${POLYFEM_INCLUDE_DIR}")

# Use C++17
target_compile_features(polyfem PUBLIC cxx_std_17)

# No limit yay
target_compile_definitions(polyfem PUBLIC -DEIGEN_STACK_ALLOCATION_LIMIT=0)

# 8MB
# target_compile_definitions(polyfem PUBLIC -DEIGEN_STACK_ALLOCATION_LIMIT=8388608)

# Max stack-size small vectors (for gradient and Hessian)
target_compile_definitions(polyfem PUBLIC -DPOLYFEM_SMALL_N=${POLYFEM_SMALL_N})
target_compile_definitions(polyfem PUBLIC -DPOLYFEM_BIG_N=${POLYFEM_BIG_N})

set(POLYFEM_JSON_SPEC_DIR "${PROJECT_SOURCE_DIR}/json-specs")
target_compile_definitions(polyfem PUBLIC POLYFEM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_compile_definitions(polyfem PUBLIC POLYFEM_JSON_SPEC_DIR="${POLYFEM_JSON_SPEC_DIR}")
target_compile_definitions(polyfem PUBLIC POLYFEM_INPUT_SPEC="${POLYFEM_JSON_SPEC_DIR}/input-spec.json")

if (MSVC)
    add_compile_options(/bigobj)
endif ()

################################################################################
# CUDA
################################################################################

if(IPC_TOOLKIT_WITH_CUDA OR POLYSOLVE_WITH_CUSOLVER)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
    else()
        message(FATAL_ERROR "No CUDA support found!")
    endif()

    set_target_properties(polyfem PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.24.0")
        set(CMAKE_CUDA_ARCHITECTURES "native")
        set_target_properties(ipc_toolkit PROPERTIES CUDA_ARCHITECTURES "native")
    else()
        include(FindCUDA/select_compute_arch)
        CUDA_DETECT_INSTALLED_GPUS(CUDA_ARCH_LIST)
        string(STRIP "${CUDA_ARCH_LIST}" CUDA_ARCH_LIST)
        string(REPLACE " " ";" CUDA_ARCH_LIST "${CUDA_ARCH_LIST}")
        string(REPLACE "." "" CUDA_ARCH_LIST "${CUDA_ARCH_LIST}")
        set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
        set_target_properties(ipc_toolkit PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
    endif()
endif()

################################################################################
# Required libraries
################################################################################

if(WIN32)
    # Eigen
    include(eigen)
    target_link_libraries(polyfem PUBLIC Eigen3::Eigen)
endif()

# openvdb library
include(openvdb)
target_link_libraries(polyfem PUBLIC openvdb)

# polysolve
include(polysolve)
target_link_libraries(polyfem PUBLIC polysolve::polysolve)

# libigl
include(libigl)
target_link_libraries(polyfem PUBLIC igl::core)

include(finite_diff)
target_link_libraries(polyfem PUBLIC finitediff::finitediff)

if(NOT WIN32)
    # Eigen
    include(eigen)
    target_link_libraries(polyfem PUBLIC Eigen3::Eigen)
endif()

# IPC Toolkit
# TensorBoard (optional; pulled early to control protobuf/absl)
if(POLYFEM_WITH_TENSORBOARD)
    include(tensorboard_logger)
    target_link_libraries(polyfem PUBLIC tensorboard_logger::tensorboard_logger)
    target_compile_definitions(polyfem PUBLIC -DPOLYFEM_WITH_TENSORBOARD)
endif()

include(ipc_toolkit)
target_link_libraries(polyfem PUBLIC ipc::toolkit)

# Threading
if(POLYFEM_THREADING STREQUAL "CPP")
    target_compile_definitions(polyfem PUBLIC POLYFEM_WITH_CPP_THREADS)
    # These definitions help avoid problems with GCC using TBB backend
    target_compile_definitions(polyfem PUBLIC PSTL_USE_PARALLEL_POLICIES=0)
    target_compile_definitions(polyfem PUBLIC _GLIBCXX_USE_TBB_PAR_BACKEND=0)
elseif(POLYFEM_THREADING STREQUAL "TBB")
    include(onetbb)
    target_link_libraries(polyfem PUBLIC TBB::tbb)
    target_compile_definitions(polyfem PUBLIC -DPOLYFEM_WITH_TBB)
endif()

# paraviewo (includes HDF5 and h5pp)
if(POLYFEM_WITH_PARAVIEWO)
    include(paraviewo)
    target_link_libraries(polyfem PUBLIC paraviewo::paraviewo)
    target_compile_definitions(polyfem PUBLIC -DPOLYFEM_WITH_PARAVIEWO)
endif()

# spdlog
include(spdlog)
target_link_libraries(polyfem PUBLIC spdlog::spdlog)

# JSON Specification Engine library
include(jse)
target_link_libraries(polyfem PUBLIC jse::jse)

# Link libdl on Linux for static MKL and other dl* users
if(UNIX AND NOT APPLE)
    target_link_libraries(polyfem PUBLIC dl)
endif()

################################################################################
# Optional libraries
################################################################################

# Extra warnings (link this here so it has top priority)
include(polyfem_warnings)
target_link_libraries(polyfem PRIVATE polyfem::warnings)

################################################################################
# Polyfem binary
################################################################################

# Main executable
if(POLYFEM_TOPLEVEL_PROJECT)
    add_executable(${PROJECT_NAME}_bin src/polyfem/main.cpp)

    target_compile_features(${PROJECT_NAME}_bin PUBLIC cxx_std_14)

    target_link_libraries(${PROJECT_NAME}_bin PUBLIC polyfem::polyfem polyfem::warnings)

    include(cli11)
    target_link_libraries(${PROJECT_NAME}_bin PUBLIC CLI11::CLI11)

    if(NOT (${CMAKE_VERSION} VERSION_LESS "3.6.0"))
        set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME}_bin)
    endif()

    if(UNIX AND NOT APPLE)
        target_link_libraries(${PROJECT_NAME}_bin PRIVATE dl)
    endif()
endif()

################################################################################
# Tests
################################################################################

# Compile extras only if this is a top-level project
if(POLYFEM_TOPLEVEL_PROJECT AND POLYFEM_WITH_TESTS)
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif()

