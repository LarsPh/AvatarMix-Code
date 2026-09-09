set(SOURCES
	autodiff.h
	DisableWarnings.hpp
	EnableWarnings.hpp
	getRSS.c
	JSONUtils.cpp
	JSONUtils.hpp
	Logger.cpp
	Logger.hpp
	MatrixCache.cpp
	MatrixCache.hpp
	MatrixUtils.cpp
	MatrixUtils.hpp
	ChromeTrace.cpp
	ChromeTrace.hpp
	MaybeParallelFor.hpp
	MaybeParallelFor.tpp
	par_for.cpp
	par_for.hpp
	StringUtils.cpp
	StringUtils.hpp
	TensorboardLogger.cpp
	TensorboardLogger.hpp
	TimingRegistry.cpp
	TimingRegistry.hpp
	Timer.hpp
	Types.hpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
