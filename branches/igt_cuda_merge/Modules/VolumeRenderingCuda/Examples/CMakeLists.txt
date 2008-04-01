Project(VolumeRenderingCudaExamples)

ADD_EXECUTABLE(VolumeRenderingCudaExample
  VolumeRenderingCudaExample.cxx
)
TARGET_LINK_LIBRARIES(VolumeRenderingCudaExample
  VolumeRenderingCudaModule
)






####################
#### ULTRASOUND ####
####################
SET(ULTRASOUND_EXAMPLE_SRCS
  vtkUltrasoundGUI.cxx
)

## WRAPPING
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE("${VTK_CMAKE_DIR}/vtkWrapTcl.cmake")
VTK_WRAP_TCL3(UltrasoundLibrary
         ULTRASOUND_EXAMPLE_TCL_SRCS 
         "${ULTRASOUND_EXAMPLE_SRCS}" 
         "")

ADD_LIBRARY(UltrasoundLibrary
  ${ULTRASOUND_EXAMPLE_SRCS}
  ${ULTRASOUND_EXAMPLE_TCL_SRCS}
)

TARGET_LINK_LIBRARIES(UltrasoundLibrary
  vtkITK
  vtkCommonTCL
  vtkImagingTCL
  vtkFilteringTCL
  vtkIOTCL
  ${KWWidgets_LIBRARIES}
)

IF(BUILD_SHARED_LIBS)
  INSTALL(TARGETS UltrasoundLibrary
      RUNTIME DESTINATION bin COMPONENT RuntimeLibraries 
      LIBRARY DESTINATION lib COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION lib COMPONENT Development)
ENDIF(BUILD_SHARED_LIBS)


ADD_EXECUTABLE(UltrasoundHeartExample
  UltrasoundHeartExample.cxx
)

TARGET_LINK_LIBRARIES(UltrasoundHeartExample
  UltrasoundLibrary
  ${KWWidgets_LIBRARIES}
)

INCLUDE_DIRECTORIES(
 ${VolumeRenderingModule_SOURCE_DIR}
 ${VolumeRenderingModule_BINARY_DIR}
  ${CudaSupport_SOURCE_DIR}
  ${CudaSupport_BINARY_DIR}
)


