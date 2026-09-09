set(SOURCES
	Common.hpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})

################################################################################
# Subfolders
################################################################################

add_subdirectory(autogen)
add_subdirectory(io)
add_subdirectory(mesh)
add_subdirectory(solver)
add_subdirectory(utils)
add_subdirectory(garment)
