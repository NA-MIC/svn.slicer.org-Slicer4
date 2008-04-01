PROJECT(UltrasoundModule)

OPTION(ULTRASOUNDMODULE "Would you like to build Ultrasound Module support" ON)
if(ULTRASOUNDMODULE)


# --------------------------------------------------------------------------
# Sources
  SET(UltrasoundModule_SRCS
    vtkUltrasoundModuleGUI.cxx
    vtkUltrasoundModuleLogic.cxx
    
    vtkMRMLUltrasoundNode.cxx
)

#-----------------------------------------------
# Include dirs

INCLUDE_DIRECTORIES(
  ${UltrasoundModule_SOURCE_DIR}
  ${UltrasoundModule_BINARY_DIR}
  ${SlicerBase_SOURCE_DIR}
  ${SlicerBase_BINARY_DIR}
  ${SlicerBaseLogic_SOURCE_DIR}
  ${SlicerBaseLogic_BINARY_DIR}
  ${SlicerBaseGUI_SOURCE_DIR}
  ${SlicerBaseGUI_BINARY_DIR}
  ${vtkITK_SOURCE_DIR}
  ${vtkITK_BINARY_DIR}
  ${ITK_INCLUDE_DIRS}
  ${VTK_INCLUDE_DIRS}
  ${MRML_SOURCE_DIR}
  ${MRML_BINARY_DIR}
  )


# --------------------------------------------------------------------------
# Wrapping

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE("${VTK_CMAKE_DIR}/vtkWrapTcl.cmake")
VTK_WRAP_TCL3(UltrasoundModule
         UltrasoundModule_TCL_SRCS 
         "${UltrasoundModule_SRCS}" 
         "")


# --------------------------------------------------------------------------
# Build the library

## Set the Libraries we link
SET(UltrasoundModule_LIBRARIES
  SlicerBaseLogic 
  SlicerBaseGUI 
  MRML
  vtkITK
  vtkCommonTCL
  vtkImagingTCL
  vtkFilteringTCL
  vtkIOTCL
  ${KWWidgets_LIBRARIES}
 )


IF(BUILD_SHARED_LIBS)
  INSTALL(TARGETS UltrasoundModule
      RUNTIME DESTINATION bin COMPONENT RuntimeLibraries 
      LIBRARY DESTINATION lib COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION lib COMPONENT Development)
ENDIF(BUILD_SHARED_LIBS)

TARGET_LINK_LIBRARIES(UltrasoundModule
  ${UltrasoundModule_LIBRARIES}
  )


# --------------------------------------------------------------------------
# Testing (requires some of the examples)

#IF(BUILD_TESTING)
#  ADD_SUBDIRECTORY(Testing)
#ENDIF(BUILD_TESTING)

##IF(BUILD_EXAMPLES)
#  ADD_SUBDIRECTORY(Examples)
##ENDIF(BUILD_EXAMPLES)

CONFIGURE_FILE(
  ${UltrasoundModule_SOURCE_DIR}/vtkUltrasoundModuleConfigure.h.in 
  ${UltrasoundModule_BINARY_DIR}/vtkUltrasoundModuleConfigure.h
)
CONFIGURE_FILE(
  ${UltrasoundModule_SOURCE_DIR}/presets.xml
  ${UltrasoundModule_BINARY_DIR}/presets.xml
  COPYONLY
)
ENDIF(ULTRASOUNDMODULE)
