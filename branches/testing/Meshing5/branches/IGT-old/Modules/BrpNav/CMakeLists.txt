PROJECT(BrpNav)

IF(NOT USE_IGSTK)
 ADD_DEFINITIONS(-DIGSTK_OFF)
ENDIF(NOT USE_IGSTK)

#
# Sources
#
SET(BrpNav_SRCS 
  # module
  vtkBrpNavLogic.cxx
  vtkBrpNavGUI.cxx
  )

#
# Include dirs
#
MESSAGE("\n\n navitrack inc = ${NAVITRACK_INC}")

INCLUDE_DIRECTORIES(
  ${ModulesBaseClasses_SOURCE_DIR}
  ${ModulesBaseClasses_BINARY_DIR}
  ${BrpNav_SOURCE_DIR}
  ${BrpNav_BINARY_DIR}
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
  ${IGT_SOURCE_DIR}
  ${IGT_BINARY_DIR}
  ${NAVITRACK_INC}
  ${IGSTK_INCLUDE_DIRS}
  )

#
# Wrapping
#
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE("${VTK_CMAKE_DIR}/vtkWrapTcl.cmake")
VTK_WRAP_TCL3(BrpNav BrpNav_TCL_SRCS "${BrpNav_SRCS}" "")

#
# Build the library
#
ADD_LIBRARY(BrpNav
  ${BrpNav_SRCS} 
  ${BrpNav_TCL_SRCS}
  )

IF(BUILD_SHARED_LIBS)
  INSTALL_TARGETS(${SLICER3_LIB_INSTALL_DIR} BrpNav)
ENDIF(BUILD_SHARED_LIBS)

TARGET_LINK_LIBRARIES(BrpNav
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
  IGT
  ${IGSTK_LIB}
  )

#
# Configure include file
#
CONFIGURE_FILE(
  ${BrpNav_SOURCE_DIR}/vtkBrpNavConfigure.h.in 
  ${BrpNav_BINARY_DIR}/vtkBrpNavConfigure.h
  )
