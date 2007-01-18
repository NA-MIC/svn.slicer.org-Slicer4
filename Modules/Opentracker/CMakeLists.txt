PROJECT(Opentracker)

#
# Sources
#
SET(Neurosurgery_SRCS 
  # module
  vtkOpentrackerLogic.cxx
  vtkOpentrackerGUI.cxx
  )

#
# Include dirs
#
INCLUDE_DIRECTORIES(
  ${ModulesBaseClasses_SOURCE_DIR}
  ${ModulesBaseClasses_BINARY_DIR}
  ${Opentracker_SOURCE_DIR}
  ${Opentracker_BINARY_DIR}
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
  ${OPENTRACKER_INC_DIR}
  )

#
# Wrapping
#
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE("${VTK_CMAKE_DIR}/vtkWrapTcl.cmake")
VTK_WRAP_TCL3(Opentracker Opentracker_TCL_SRCS "${Opentracker_SRCS}" "")

#
# Build the library
#
ADD_LIBRARY(Opentracker
  ${Opentracker_SRCS} 
  ${Opentracker_TCL_SRCS}
  )

IF(BUILD_SHARED_LIBS)
  INSTALL_TARGETS(${SLICER3_LIB_INSTALL_DIR} Opentracker)
ENDIF(BUILD_SHARED_LIBS)

TARGET_LINK_LIBRARIES(Opentracker
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
  vtkHybridTCL
  ${OPENTRACKER_LIB}
  IGT

  )

#
# Configure include file
#
CONFIGURE_FILE(
  ${Opentracker_SOURCE_DIR}/vtkOpentrackerConfigure.h.in 
  ${Opentracker_BINARY_DIR}/vtkOpentrackerConfigure.h
  )
