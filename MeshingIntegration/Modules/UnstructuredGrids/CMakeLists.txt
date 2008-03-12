PROJECT(UnstructuredGrids)

# --------------------------------------------------------------------------
# Sources

SET(UnstructuredGrids_SRCS
  vtkSlicerUnstructuredGridsLogic.cxx
  vtkSlicerUnstructuredGridsGUI.cxx
  #vtkMRMLUnstructuredGridsNode.cxx
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
  ${UnstructuredGrids_SOURCE_DIR}
  ${UnstructuredGrids_BINARY_DIR}
  ${SlicerBase_SOURCE_DIR}
  ${SlicerBase_BINARY_DIR}
  ${SlicerBaseLogic_SOURCE_DIR}
  ${SlicerBaseLogic_BINARY_DIR}
  ${SlicerBaseGUI_SOURCE_DIR}
  ${SlicerBaseGUI_BINARY_DIR}
  ${MRML_SOURCE_DIR}
  ${MRML_BINARY_DIR}
  )


# --------------------------------------------------------------------------
# Wrapping

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE("${VTK_CMAKE_DIR}/vtkWrapTcl.cmake")
VTK_WRAP_TCL3(UnstructuredGrids UnstructuredGrids_TCL_SRCS "${UnstructuredGrids_SRCS}" "")


# --------------------------------------------------------------------------
# Build the library

ADD_LIBRARY(UnstructuredGrids
  ${UnstructuredGrids_SRCS}
  ${UnstructuredGrids_TCL_SRCS}
  )

IF(BUILD_SHARED_LIBS)
  INSTALL(TARGETS UnstructuredGrids
      RUNTIME DESTINATION bin COMPONENT RuntimeLibraries 
      LIBRARY DESTINATION lib COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION lib COMPONENT Development)
ENDIF(BUILD_SHARED_LIBS)

TARGET_LINK_LIBRARIES(UnstructuredGrids
  SlicerBaseLogic 
  SlicerBaseGUI 
  MRML
  vtkCommonTCL
  vtkFilteringTCL
  vtkIOTCL
  ${KWWidgets_LIBRARIES} )



# --------------------------------------------------------------------------
# Testing (requires some of the examples)

IF(BUILD_TESTING)
  SUBDIRS(Testing)
ENDIF(BUILD_TESTING)

CONFIGURE_FILE(
  ${UnstructuredGrids_SOURCE_DIR}/vtkSlicerUnstructuredGridsConfigure.h.in 
  ${UnstructuredGrids_BINARY_DIR}/vtkSlicerUnstructuredGridsConfigure.h
)
