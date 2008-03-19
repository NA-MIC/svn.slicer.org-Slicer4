PROJECT(OpenIGTLinkDaemon)

# --------------------------------------------------------------------------
# Sources

SET(OpenIGTLinkDaemon_SRCS
  #vtkOpenIGTLinkDaemonLogic.cxx
  #vtkOpenIGTLinkDaemonGUI.cxx
  #vtkMRMLOpenIGTLinkDaemonNode.cxx
  vtkOpenIGTLinkTclHelper.cxx
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

# Abstract/pure virtual classes

#SET_SOURCE_FILES_PROPERTIES(
  #vtkModule.cxx
  #PROPERTIES
  #ABSTRACT "TRUE"
  #)

# Helper classes

#SET_SOURCE_FILES_PROPERTIES(
  #vtkModule.cxx
  #PROPERTIES
  #WRAP_EXCLUDE "TRUE"
  #)

# --------------------------------------------------------------------------
# Include dirs

INCLUDE_DIRECTORIES(
  ${OpenIGTLinkDaemon_SOURCE_DIR}
  ${OpenIGTLinkDaemon_SOURCE_DIR}/igtlutil
  ${OpenIGTLinkDaemon_BINARY_DIR}
  ${SlicerBase_SOURCE_DIR}
  ${SlicerBase_BINARY_DIR}
  ${SlicerBaseLogic_SOURCE_DIR}
  ${SlicerBaseLogic_BINARY_DIR}
  ${SlicerBaseGUI_SOURCE_DIR}
  ${SlicerBaseGUI_BINARY_DIR}
  ${VTK_INCLUDE_DIR}
  ${MRML_SOURCE_DIR}
  ${MRML_BINARY_DIR}
  )


# --------------------------------------------------------------------------
# Wrapping

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE("${VTK_CMAKE_DIR}/vtkWrapTcl.cmake")
VTK_WRAP_TCL3(OpenIGTLinkDaemon OpenIGTLinkDaemon_TCL_SRCS "${OpenIGTLinkDaemon_SRCS}" "")


# --------------------------------------------------------------------------
# Build the library

ADD_LIBRARY(OpenIGTLinkDaemon
  ${OpenIGTLinkDaemon_SRCS}
  ${OpenIGTLinkDaemon_TCL_SRCS}
  ${OpenIGTLinkUtil_SRCS}
  )

IF(BUILD_SHARED_LIBS)
  INSTALL_TARGETS(${SLICER3_LIB_INSTALL_DIR} OpenIGTLinkDaemon)
ENDIF(BUILD_SHARED_LIBS)

TARGET_LINK_LIBRARIES(OpenIGTLinkDaemon
  SlicerBaseLogic 
  SlicerBaseGUI 
  MRML
  vtkCommonTCL
  vtkImagingTCL
  ${KWWidgets_LIBRARIES} )



# --------------------------------------------------------------------------
# Testing (requires some of the examples)

#IF(BUILD_TESTING)
#  SUBDIRS(Testing)
#ENDIF(BUILD_TESTING)

#
# configure files to drop in the build directory
#
CONFIGURE_FILE(
  ${OpenIGTLinkDaemon_SOURCE_DIR}/vtkOpenIGTLinkDaemonConfigure.h.in 
  ${OpenIGTLinkDaemon_BINARY_DIR}/vtkOpenIGTLinkDaemonConfigure.h
)

CONFIGURE_FILE(
  ${OpenIGTLinkDaemon_SOURCE_DIR}/Tcl/openigtlinkd.tcl
  ${SLICER_LIBRARIES_DIR}/openigtlinkd.tcl
)

#CONFIGURE_FILE(
#  ${OpenIGTLinkDaemon_SOURCE_DIR}/Tcl/slicerget.tcl
#  ${SLICER_LIBRARIES_DIR}/slicerget.tcl
#)
#
#CONFIGURE_FILE(
#  ${OpenIGTLinkDaemon_SOURCE_DIR}/Tcl/slicerput.tcl
#  ${SLICER_LIBRARIES_DIR}/slicerput.tcl
#)

INSTALL(FILES
  "${SLICER_LIBRARIES_DIR}/openigtlinkd.tcl"
  DESTINATION ${SLICER_INSTALL_LIBRARIES_DIR})

#INSTALL(FILES
#  "${SLICER_LIBRARIES_DIR}/slicerput.tcl"
#  DESTINATION ${SLICER_INSTALL_LIBRARIES_DIR})
#
#INSTALL(FILES
#  "${SLICER_LIBRARIES_DIR}/slicerget.tcl"
#  DESTINATION ${SLICER_INSTALL_LIBRARIES_DIR})
#

