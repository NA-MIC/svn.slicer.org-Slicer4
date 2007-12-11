PROJECT(VolumeRenderingCudaModule)
OPTION(VOLUMERENDERINGCUDAMODULE "Would you like to build Volume Rendering with CUDA support" OFF)
if(VOLUMERENDERINGCUDAMODULE)

INCLUDE(${VolumeRenderingCudaModule_SOURCE_DIR}/cuda/FindCuda.cmake)

# --------------------------------------------------------------------------
# Sources
  SET(VolumeRenderingCudaModule_SRCS
    vtkCudaSupport.cxx
    vtkCudaDevice.cxx
    vtkVolumeCudaMapper.cxx
    vtkVolumeRenderingCudaFactory.cxx
    vtkVolumeRenderingCudaGUI.cxx
  )
  SET(VolumeRenderingCudaModule_CUDA_SRCS
  )

#-----------------------------------------------
# Include dirs

INCLUDE_DIRECTORIES(
  ${FOUND_CUT_INCLUDE}
  ${CUDA_INSTALL_PREFIX}/include
  ${VolumeRenderingCudaModule_SOURCE_DIR}
  ${VolumeRenderingCudaModule_BINARY_DIR}
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

 CUDA_INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR} 
                           ${FOUND_CUT_INCLUDE}
 )

# --------------------------------------------------------------------------
# Wrapping

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE("${VTK_CMAKE_DIR}/vtkWrapTcl.cmake")
VTK_WRAP_TCL3(VolumeRenderingCudaModule
         VolumeRenderingCudaModule_TCL_SRCS 
         "${VolumeRenderingCudaModule_SRCS}" 
         "")


# --------------------------------------------------------------------------
# Build the library

## Set the Libraries we link
SET(VolumeRenderingCudaModule_LIBRARIES
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
 )

## Build the Cuda Library first
CUDA_ADD_LIBRARY(VolumeRenderingCudaModule_CudaLibrary
  ${VolumeRenderingCudaModule_CUDA_SRCS}
  )

ADD_LIBRARY(VolumeRenderingCudaModule
  ${VolumeRenderingCudaModule_SRCS}
  ${VolumeRenderingCudaModule_TCL_SRCS}
  )

IF(BUILD_SHARED_LIBS)
  INSTALL(TARGETS VolumeRenderingCudaModule
      RUNTIME DESTINATION bin COMPONENT RuntimeLibraries 
      LIBRARY DESTINATION lib COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION lib COMPONENT Development)
ENDIF(BUILD_SHARED_LIBS)

TARGET_LINK_LIBRARIES(VolumeRenderingCudaModule
  ${VolumeRenderingCudaModule_LIBRARIES}
  ${VolumeRenderingCudaModule_CudaLibrary}
  )


##########################
## TO BE DELETED AGAIN ###
##########################
# CUDA_ADD_LIBRARY( test_lib
#${VolumeRenderingCudaModule_CUDA_SRCS}
# )
#
# CUDA_ADD_EXECUTABLE(lib_example
#${VolumeRenderingCudaModule_CUDA_SRCS}
#${VolumeRenderingCudaModule_SRCS}
#main.cpp
# )
#
# CUDA_ADD_LIBRARY( test_lib
#${VolumeRenderingCudaModule_CUDA_SRCS}
#${VolumeRenderingCudaModule_SRCS}
#)
#
# TARGET_LINK_LIBRARIES(lib_example
##       test_lib
#GL
#GLU
#${FOUND_CUT}
#${VolumeRenderingCudaModule_LIBRARIES}
# )
##########################


# --------------------------------------------------------------------------
# Testing (requires some of the examples)

#IF(BUILD_TESTING)
#  SUBDIRS(Testing)
#ENDIF(BUILD_TESTING)

CONFIGURE_FILE(
  ${VolumeRenderingCudaModule_SOURCE_DIR}/vtkVolumeRenderingCudaModuleConfigure.h.in 
  ${VolumeRenderingCudaModule_BINARY_DIR}/vtkVolumeRenderingCudaModuleConfigure.h
)
CONFIGURE_FILE(
  ${VolumeRenderingCudaModule_SOURCE_DIR}/presets.xml
  ${VolumeRenderingCudaModule_BINARY_DIR}/presets.xml
  COPYONLY
)
ENDIF(VOLUMERENDERINGCUDAMODULE)
