#
# Protobuf (https://github.com/protocolbuffers/protobuf)
# License: BSD 3-Clause
#

if(TARGET protobuf::libprotobuf)
    return()
endif()

message(STATUS "Third-party: creating target 'protobuf::libprotobuf'")

include(CPM)

# Abseil (required by recent Protobuf)
# Protobuf v25+ includes absl/base/prefetch.h which is not present in older Abseil releases.
if(NOT TARGET absl::strings)
    set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)
    set(ABSL_MSVC_STATIC_RUNTIME OFF CACHE BOOL "" FORCE)
    CPMAddPackage(
        NAME abseil-cpp
        GITHUB_REPOSITORY abseil/abseil-cpp
        GIT_TAG 20230802.0
    )
    set(ABSL_ROOT_DIR "${abseil-cpp_SOURCE_DIR}" CACHE PATH "" FORCE)
endif()

# Keep protobuf build minimal
set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(protobuf_INSTALL OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(protobuf_WITH_ZLIB OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_PROTOC_BINARIES ON CACHE BOOL "" FORCE)
set(protobuf_ABSL_PROVIDER "module" CACHE STRING "" FORCE)
set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE BOOL "" FORCE)

CPMAddPackage(
    NAME protobuf
    GITHUB_REPOSITORY protocolbuffers/protobuf
    GIT_TAG v25.3
)

if(TARGET libprotobuf AND NOT TARGET protobuf::libprotobuf)
    add_library(protobuf::libprotobuf ALIAS libprotobuf)
endif()

if(TARGET protoc AND NOT TARGET protobuf::protoc)
    add_executable(protobuf::protoc ALIAS protoc)
endif()

set_target_properties(libprotobuf PROPERTIES FOLDER external)

if(TARGET protoc)
    set_target_properties(protoc PROPERTIES FOLDER external)
endif()

