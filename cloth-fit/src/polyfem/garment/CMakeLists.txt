set(SOURCES
	optimize.cpp
	optimize.hpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfem PUBLIC ${SOURCES})
