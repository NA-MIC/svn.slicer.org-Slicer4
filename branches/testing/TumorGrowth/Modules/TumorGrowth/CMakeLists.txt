PROJECT(TumorGrowth)

#
# Sources
#
SET(TumorGrowth_SRCS 
  # module
  vtkTumorGrowthGUI.cxx
  vtkTumorGrowthLogic.cxx 
  # Filters
  ${CMAKE_CURRENT_SOURCE_DIR}/Filters/vtkImageRectangularSource.cxx
  ${CMAKE_CURRENT_SOURCE_DIR}/Filters/vtkImageIslandFilter.cxx
  ${CMAKE_CURRENT_SOURCE_DIR}/Filters/vtkImageHistogramNormalization.cxx
  ${CMAKE_CURRENT_SOURCE_DIR}/Filters/vtkImageGCR.cxx
  ${CMAKE_CURRENT_SOURCE_DIR}/Filters/vtkImageKilianDistanceTransform.cxx
  ${CMAKE_CURRENT_SOURCE_DIR}/Filters/vtkImageSumOverVoxels.cxx 
  # wizard
  ${CMAKE_CURRENT_SOURCE_DIR}/Wizard/vtkTumorGrowthStep.cxx
  ${CMAKE_CURRENT_SOURCE_DIR}/Wizard/vtkTumorGrowthSelectScanStep.cxx
  ${CMAKE_CURRENT_SOURCE_DIR}/Wizard/vtkTumorGrowthFirstScanStep.cxx
  ${CMAKE_CURRENT_SOURCE_DIR}/Wizard/vtkTumorGrowthROIStep.cxx
  ${CMAKE_CURRENT_SOURCE_DIR}/Wizard/vtkTumorGrowthSegmentationStep.cxx
  # ${CMAKE_CURRENT_SOURCE_DIR}/Wizard/vtkTumorGrowthSecondScanStep.cxx
  ${CMAKE_CURRENT_SOURCE_DIR}/Wizard/vtkTumorGrowthTypeStep.cxx
  ${CMAKE_CURRENT_SOURCE_DIR}/Wizard/vtkTumorGrowthAnalysisStep.cxx
  # MRML
  ${CMAKE_CURRENT_SOURCE_DIR}/vtkMRMLTumorGrowthNode.cxx
)

# SET_SOURCE_FILES_PROPERTIES(
#  # Algorithm sources (temporary exclusion)
#  WRAP_EXCLUDE
#  )

#
# Include dirs
#
INCLUDE_DIRECTORIES(
  ${TumorGrowth_SOURCE_DIR}
  ${TumorGrowth_BINARY_DIR}
  ${TumorGrowth_SOURCE_DIR}/Wizard
  ${TumorGrowth_SOURCE_DIR}/Filters
  ${SlicerBase_SOURCE_DIR}
  ${SlicerBase_BINARY_DIR}
  ${SlicerBaseLogic_SOURCE_DIR}
  ${SlicerBaseLogic_BINARY_DIR}
  ${SlicerBaseGUI_SOURCE_DIR}
  ${SlicerBaseGUI_BINARY_DIR}
  ${vtkITK_SOURCE_DIR}
  ${vtkITK_BINARY_DIR}
  ${ITK_INCLUDE_DIR}
  ${VTK_INCLUDE_DIR}
  ${MRML_SOURCE_DIR}
  ${MRML_BINARY_DIR}
  )

#
# Wrapping
#
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE("${VTK_CMAKE_DIR}/vtkWrapTcl.cmake")
VTK_WRAP_TCL3(TumorGrowth TumorGrowth_TCL_SRCS "${TumorGrowth_SRCS}" "")

#
# Build the library
#
ADD_LIBRARY(TumorGrowth 
  ${TumorGrowth_SRCS} 
  ${TumorGrowth_TCL_SRCS}
  )

IF(BUILD_SHARED_LIBS)
  INSTALL(TARGETS TumorGrowth
      RUNTIME DESTINATION bin COMPONENT RuntimeLibraries 
      LIBRARY DESTINATION lib COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION lib COMPONENT Development)
ENDIF(BUILD_SHARED_LIBS)

TARGET_LINK_LIBRARIES(TumorGrowth
  SlicerBaseLogic 
  SlicerBaseGUI 
  MRML
  vtkITK
  vtkCommonTCL
  vtkImagingTCL
  vtkFilteringTCL
  vtkIOTCL
  ITKAlgorithms
  ITKNumerics 
  ITKCommon
  ITKBasicFilters
  ITKNumerics
  ITKStatistics
  ITKBasicFilters
  ITKIO
  ITKDICOMParser
  ${KWWidgets_LIBRARIES} 
  )

#
# Testing
#
IF(BUILD_TESTING)
  SUBDIRS(Testing)
ENDIF(BUILD_TESTING)

#
# command line application
#
# SUBDIRS(CommandLineApplication)

#
# Configure include file
#
CONFIGURE_FILE(
  ${TumorGrowth_SOURCE_DIR}/vtkTumorGrowthConfigure.h.in 
  ${TumorGrowth_BINARY_DIR}/vtkTumorGrowthConfigure.h
  )

#
# copy over the tcl files
#
SET (TCL_DIR ${TumorGrowth_BINARY_DIR}/tcl)
IF (NOT EXISTS ${TCL_DIR})
  FILE(MAKE_DIRECTORY ${TCL_DIR})
ENDIF (NOT EXISTS ${TCL_DIR})

SET(TCLFILES
  TumorGrowthFct.tcl 
  TumorGrowthReg.tcl
)

FOREACH(tclfile ${TCLFILES})
  CONFIGURE_FILE(
    ${Slicer3_SOURCE_DIR}/Modules/TumorGrowth/tcl/${tclfile}
    ${TCL_DIR}/${tclfile}
    COPYONLY
  )
ENDFOREACH(tclfile)
