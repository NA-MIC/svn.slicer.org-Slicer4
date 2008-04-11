Project(UltrasoundModuleExamples)

####################
#### ULTRASOUND ####
####################
SET(ULTRASOUND_EXAMPLE_SRCS
  vtkUltrasoundExampleGUI.cxx
)

## WRAPPING
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE("${VTK_CMAKE_DIR}/vtkWrapTcl.cmake")
VTK_WRAP_TCL3(UltrasoundExampleGUILibrary
         ULTRASOUND_EXAMPLE_TCL_SRCS 
         "${ULTRASOUND_EXAMPLE_SRCS}" 
         "")

ADD_LIBRARY(UltrasoundExampleGUILibrary
  ${ULTRASOUND_EXAMPLE_SRCS}
  ${ULTRASOUND_EXAMPLE_TCL_SRCS}
)

TARGET_LINK_LIBRARIES(UltrasoundExampleGUILibrary
  vtkITK
  vtkCommonTCL
  vtkImagingTCL
  vtkFilteringTCL
  vtkIOTCL
  ${KWWidgets_LIBRARIES}
)

IF(BUILD_SHARED_LIBS)
  INSTALL(TARGETS UltrasoundExampleGUILibrary
      RUNTIME DESTINATION bin COMPONENT RuntimeLibraries 
      LIBRARY DESTINATION lib COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION lib COMPONENT Development)
ENDIF(BUILD_SHARED_LIBS)


ADD_EXECUTABLE(UltrasoundHeartExample
  UltrasoundHeartExample.cxx
)

TARGET_LINK_LIBRARIES(UltrasoundHeartExample
  UltrasoundExampleGUILibrary
  ${KWWidgets_LIBRARIES}
)

INCLUDE_DIRECTORIES(
 ${VolumeRenderingModule_SOURCE_DIR}
 ${VolumeRenderingModule_BINARY_DIR}
 ${VolumeRenderingCudaModule_SOURCE_DIR}
 ${VolumeRenderingCudaModule_BINARY_DIR}
 ${CudaSupport_SOURCE_DIR}
 ${CudaSupport_BINARY_DIR}
)
