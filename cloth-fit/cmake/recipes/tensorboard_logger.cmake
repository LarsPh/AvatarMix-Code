#
# tensorboard_logger (https://github.com/RustingSword/tensorboard_logger)
# License: MIT
#

if(TARGET tensorboard_logger::tensorboard_logger OR TARGET tensorboard_logger)
    return()
endif()

message(STATUS "Third-party: creating target 'tensorboard_logger::tensorboard_logger'")

include(CPM)
include(protobuf)

CPMAddPackage(
    NAME tensorboard_logger_src
    GITHUB_REPOSITORY RustingSword/tensorboard_logger
    GIT_TAG master
    DOWNLOAD_ONLY YES
)

set(TB_SRC_DIR "${tensorboard_logger_src_SOURCE_DIR}")
set(TB_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/tensorboard_logger_proto")
file(MAKE_DIRECTORY "${TB_GEN_DIR}")

file(GLOB TB_PROTOS "${TB_SRC_DIR}/proto/*.proto")

# Protobuf "well-known types" (e.g. google/protobuf/struct.proto) live under
# <protobuf>/src, so add that to protoc include paths.
set(TB_PROTO_IMPORT_DIRS "${TB_SRC_DIR}/proto")
if(DEFINED protobuf_SOURCE_DIR AND EXISTS "${protobuf_SOURCE_DIR}/src/google/protobuf/struct.proto")
    list(APPEND TB_PROTO_IMPORT_DIRS "${protobuf_SOURCE_DIR}/src")
endif()
if(DEFINED protobuf_BINARY_DIR AND EXISTS "${protobuf_BINARY_DIR}")
    list(APPEND TB_PROTO_IMPORT_DIRS "${protobuf_BINARY_DIR}")
endif()

set(TB_PROTO_INCLUDE_ARGS "")
foreach(dir IN LISTS TB_PROTO_IMPORT_DIRS)
    list(APPEND TB_PROTO_INCLUDE_ARGS "-I" "${dir}")
endforeach()

set(TB_PROTO_SRCS "")
set(TB_PROTO_HDRS "")

foreach(proto_file IN LISTS TB_PROTOS)
    get_filename_component(proto_name "${proto_file}" NAME_WE)
    set(gen_cc "${TB_GEN_DIR}/${proto_name}.pb.cc")
    set(gen_h  "${TB_GEN_DIR}/${proto_name}.pb.h")

    add_custom_command(
        OUTPUT "${gen_cc}" "${gen_h}"
        COMMAND $<TARGET_FILE:protobuf::protoc>
            "--cpp_out=${TB_GEN_DIR}"
            ${TB_PROTO_INCLUDE_ARGS}
            "${proto_file}"
        DEPENDS "${proto_file}" protobuf::protoc
        COMMENT "Generating protobuf ${proto_name}.pb.cc/.h"
        VERBATIM
    )

    list(APPEND TB_PROTO_SRCS "${gen_cc}")
    list(APPEND TB_PROTO_HDRS "${gen_h}")
endforeach()

add_library(tensorboard_logger
    "${TB_SRC_DIR}/src/crc.cc"
    "${TB_SRC_DIR}/src/tensorboard_logger.cc"
    ${TB_PROTO_SRCS}
)

add_library(tensorboard_logger::tensorboard_logger ALIAS tensorboard_logger)

target_compile_features(tensorboard_logger PUBLIC cxx_std_17)

target_include_directories(tensorboard_logger
PUBLIC
    "${TB_SRC_DIR}/include"
    "${TB_GEN_DIR}"
)

target_link_libraries(tensorboard_logger PUBLIC protobuf::libprotobuf)

set_target_properties(tensorboard_logger PROPERTIES FOLDER external)

