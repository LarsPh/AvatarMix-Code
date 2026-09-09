set(SOURCES
	GarmentForm.cpp
	GarmentForm.hpp
	GarmentALForm.cpp
	GarmentALForm.hpp
	CurveConstraintForm.cpp
	CurveConstraintForm.hpp
	CurveCenterTargetForm.cpp
	CurveCenterTargetForm.hpp
	FitForm.cpp
	FitForm.hpp
	SkeletonInsideSDFForm.cpp
	SkeletonInsideSDFForm.hpp
	BoneLengthForm.cpp
	BoneLengthForm.hpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
