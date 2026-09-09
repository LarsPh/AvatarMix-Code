# Polyfem Solvers (https://github.com/Huangzizhou/openvdb)
# License: MIT

if(TARGET openvdb::openvdb)
    return()
endif()

message(STATUS "Third-party: creating target 'openvdb::openvdb'")

option(OPENVDB_BUILD_PYTHON_MODULE "Build the pyopenvdb Python module" OFF)
option(OPENVDB_CORE_SHARED "Build dynamically linked version of the core library." OFF)
option(OPENVDB_ENABLE_UNINSTALL "Adds a CMake uninstall target." OFF)

include(CPM)
CPMAddPackage("gh:Huangzizhou/openvdb#c96cb06971a89ad2638ed29972ab54f63a8e2fbc")
