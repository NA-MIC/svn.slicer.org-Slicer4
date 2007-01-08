PROJECT(Neurosurgery)

#
# Sources
#
SET(Neurosurgery_SRCS 
  # module
  vtkNeurosurgeryLogic.cxx
  vtkNeurosurgeryGUI.cxx
  )

#
# Include dirs
#
INCLUDE_DIRECTORIES(
  ${ModulesBaseClasses_SOURCE_DIR}
  ${ModulesBaseClasses_BINARY_DIR}
  ${Neurosurgery_SOURCE_DIR}
  ${Neurosurgery_BINARY_DIR}
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
VTK_WRAP_TCL3(Neurosurgery Neurosurgery_TCL_SRCS "${Neurosurgery_SRCS}" "")

#
# Build the library
#
ADD_LIBRARY(Neurosurgery
  ${Neurosurgery_SRCS} 
  ${Neurosurgery_TCL_SRCS}
  )

IF(BUILD_SHARED_LIBS)
  INSTALL_TARGETS(${SLICER3_LIB_INSTALL_DIR} Neurosurgery)
ENDIF(BUILD_SHARED_LIBS)

TARGET_LINK_LIBRARIES(Neurosurgery
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
  )

#
# Configure include file
#
CONFIGURE_FILE(
  ${Neurosurgery_SOURCE_DIR}/vtkNeurosurgeryConfigure.h.in 
  ${Neurosurgery_BINARY_DIR}/vtkNeurosurgeryConfigure.h
  )
