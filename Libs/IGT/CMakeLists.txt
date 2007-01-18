PROJECT(IGT)

# --------------------------------------------------------------------------
# Special build configs (to handle linking on windows)
CONFIGURE_FILE(
  ${IGT_SOURCE_DIR}/vtkIGTConfigure.h.in 
  ${IGT_BINARY_DIR}/vtkIGTConfigure.h
)


# --------------------------------------------------------------------------
# Sources

SET(IGT_SRCS
    vtkIGTDataManager.cxx
    vtkIGTDataStream.cxx
    vtkIGTMatrixState.cxx
    vtkIGTOpenTrackerStream.cxx
    vtkIGTPat2ImgRegistration.cxx
)




# --------------------------------------------------------------------------
# Include dirs

INCLUDE_DIRECTORIES(
  ${SlicerBaseGUI_SOURCE_DIR}
  ${SlicerBaseLogic_SOURCE_DIR}
  ${MRML_BINARY_DIR}
  ${MRML_SOURCE_DIR}
  ${IGT_BINARY_DIR}
  ${IGT_SOURCE_DIR}
  ${ITK_INCLUDE_DIRS}
  ${VTK_INCLUDE_DIRS}
  ${OPENTRACKER_INC_DIR}
  )



# Build the library

ADD_LIBRARY(IGT 
  ${IGT_SRCS} ${IGT_TCL_SRCS}
  )
IF(BUILD_SHARED_LIBS)
  INSTALL_TARGETS(${SLICER3_LIB_INSTALL_DIR} IGT)
ENDIF(BUILD_SHARED_LIBS)

TARGET_LINK_LIBRARIES(IGT
  MRML
  vtkCommonTCL
  vtkImagingTCL
  vtkFilteringTCL
  vtkIOTCL
  vtkGraphicsTCL
  vtkRenderingTCL
  ITKAlgorithms
  ITKNumerics 
  ITKCommon
  ITKBasicFilters
  ITKNumerics
  ITKStatistics
  ITKBasicFilters
  ITKIO
  ITKDICOMParser
  ${OPENTRACKER_LIB}
  )


# --------------------------------------------------------------------------
# Testing (requires some of the examples)

IF(BUILD_TESTING)
  SUBDIRS(Testing)
ENDIF(BUILD_TESTING)

