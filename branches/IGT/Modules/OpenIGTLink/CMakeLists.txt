PROJECT(OpenIGTLink)

#
# Type Size Check for igtlutil library
#

INCLUDE (CheckTypeSize)
CHECK_TYPE_SIZE(int         CMAKE_SIZEOF_INT)
CHECK_TYPE_SIZE(long        CMAKE_SIZEOF_LONG)
CHECK_TYPE_SIZE("void*"     CMAKE_SIZEOF_VOID_P)
CHECK_TYPE_SIZE(char        CMAKE_SIZEOF_CHAR)
CHECK_TYPE_SIZE(short       CMAKE_SIZEOF_SHORT)
CHECK_TYPE_SIZE(float       CMAKE_SIZEOF_FLOAT)
CHECK_TYPE_SIZE(double      CMAKE_SIZEOF_DOUBLE)
CHECK_TYPE_SIZE("long long" CMAKE_SIZEOF_LONG_LONG)
CHECK_TYPE_SIZE("__int64"   CMAKE_SIZEOF___INT64)

ADD_DEFINITIONS(-DIGTL_SIZEOF_CHAR=${CMAKE_SIZEOF_CHAR})
ADD_DEFINITIONS(-DIGTL_SIZEOF_DOUBLE=${CMAKE_SIZEOF_DOUBLE})
ADD_DEFINITIONS(-DIGTL_SIZEOF_FLOAT=${CMAKE_SIZEOF_FLOAT})
ADD_DEFINITIONS(-DIGTL_SIZEOF_INT=${CMAKE_SIZEOF_INT})
ADD_DEFINITIONS(-DIGTL_SIZEOF_LONG=${CMAKE_SIZEOF_LONG})
ADD_DEFINITIONS(-DIGTL_SIZEOF_SHORT=${CMAKE_SIZEOF_SHORT})
ADD_DEFINITIONS(-DIGTL_SIZEOF_FLOAT=${CMAKE_SIZEOF_FLOAT})
ADD_DEFINITIONS(-DIGTL_SIZEOF_DOUBLE=${CMAKE_SIZEOF_DOUBLE})

IF(IGTL_SIZEOF_LONG_LONG)
  ADD_DEFINITIONS(-DIGTL_TYPE_USE_LONG_LONG=1)
ENDIF(IGTL_SIZEOF_LONG_LONG)

IF(IGTL_SIZEOF___INT64)
  IF(NOT IGTL_TYPE_SAME_LONG_AND___INT64)
    IF(NOT IGTL_TYPE_SAME_LONG_LONG_AND___INT64)
      ADD_DEFINITIONS(-IGTL_TYPE_USE___INT64=1)
    ENDIF(NOT IGTL_TYPE_SAME_LONG_LONG_AND___INT64)
  ENDIF(NOT IGTL_TYPE_SAME_LONG_AND___INT64)
ENDIF(IGTL_SIZEOF___INT64)




#
# Sources
#
SET(OpenIGTLink_SRCS 
  # module
  vtkOpenIGTLinkLogic.cxx
  vtkOpenIGTLinkGUI.cxx
  vtkIGTLConnector.cxx
  vtkIGTLCircularBuffer.cxx
  vtkClientSocket.cxx
  vtkServerSocket.cxx
  vtkSocket.cxx
  )

SET(OpenIGTLinkUtil_SRCS
  igtlutil/crc32.c
  igtlutil/crc32.h
  igtlutil/igtl_header.c
  igtlutil/igtl_header.h
  igtlutil/igtl_image.c
  igtlutil/igtl_image.h
  igtlutil/igtl_transform.c
  igtlutil/igtl_transform.h
  igtlutil/igtl_util.c
  igtlutil/igtl_util.h
)

#
# Include dirs
#

INCLUDE_DIRECTORIES(
  ${ModulesBaseClasses_SOURCE_DIR}
  ${ModulesBaseClasses_BINARY_DIR}
  ${OpenIGTLink_SOURCE_DIR}
  ${OpenIGTLink_BINARY_DIR}
  ${OpenIGTLink_SOURCE_DIR}/igtlutil
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
  )

#
# Wrapping
#
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE("${VTK_CMAKE_DIR}/vtkWrapTcl.cmake")
VTK_WRAP_TCL3(OpenIGTLink OpenIGTLink_TCL_SRCS "${OpenIGTLink_SRCS}" "")

#
# Build the library
#
ADD_LIBRARY(OpenIGTLink
  ${OpenIGTLink_SRCS} 
  ${OpenIGTLink_TCL_SRCS}
  ${OpenIGTLinkUtil_SRCS}
  )

IF(BUILD_SHARED_LIBS)
  INSTALL_TARGETS(${SLICER3_LIB_INSTALL_DIR} OpenIGTLink)
ENDIF(BUILD_SHARED_LIBS)

TARGET_LINK_LIBRARIES(OpenIGTLink
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
  )

#
# Configure include file
#
CONFIGURE_FILE(
  ${OpenIGTLink_SOURCE_DIR}/vtkOpenIGTLinkConfigure.h.in 
  ${OpenIGTLink_BINARY_DIR}/vtkOpenIGTLinkConfigure.h
  )

