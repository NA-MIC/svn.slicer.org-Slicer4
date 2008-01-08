PROJECT(VolumeRenderingCudaModule)

## ONLY BUILD THIS IF YOU WANT CUDA SUPPORT TO BE ENABLED
IF(CUDA_SUPPORT_ENABLED)

OPTION(VOLUMERENDERINGCUDAMODULE "Would you like to build Volume Rendering with CUDA support" ON)
if(VOLUMERENDERINGCUDAMODULE)

INCLUDE(${CMAKE_SOURCE_DIR}/CMake/cuda/FindCuda.cmake)

# --------------------------------------------------------------------------
# Sources
  SET(VolumeRenderingCudaModule_SRCS

  ### CUDA Volume Rendering Integration Classes ###
    vtkVolumeCudaMapper.cxx    
    vtkVolumeRenderingCudaFactory.cxx
    vtkVolumeRenderingCudaModuleGUI.cxx
    vtkVolumeRenderingCudaModuleLogic.cxx
    
  ## WIDGETS ##
    vtkKWTypeChooserBox.cxx
  )
  SET(VolumeRenderingCudaModule_CUDA_SRCS
      CUDA_renderAlgo.cu
  )

#-----------------------------------------------
# Include dirs

INCLUDE_DIRECTORIES(
  ${FOUND_CUT_INCLUDE}
  ${CUDA_INSTALL_PREFIX}/include
  ${CudaSupport_SOURCE_DIR}
  ${CudaSupport_BINARY_DIR}
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

## Build the Cuda Library first
#CUDA_ADD_LIBRARY(VolumeRenderingCudaModule_CudaLibrary  
#  )
  
#TARGET_LINK_LIBRARIES(VolumeRenderingCudaModule_CudaLibrary
#  ${CudaSupport}
#)

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
  ${KWWidgets_LIBRARIES}
  ${CUDA_TARGET_LINK}
#  ${VolumeRenderingCudaModule_CudaLibrary}
  CudaSupport
 )

CUDA_ADD_LIBRARY(VolumeRenderingCudaModule
  ${VolumeRenderingCudaModule_SRCS}
  ${VolumeRenderingCudaModule_TCL_SRCS}
  ${VolumeRenderingCudaModule_CUDA_SRCS}
  )

IF(BUILD_SHARED_LIBS)
  INSTALL(TARGETS VolumeRenderingCudaModule
      RUNTIME DESTINATION bin COMPONENT RuntimeLibraries 
      LIBRARY DESTINATION lib COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION lib COMPONENT Development)
ENDIF(BUILD_SHARED_LIBS)

TARGET_LINK_LIBRARIES(VolumeRenderingCudaModule
  ${VolumeRenderingCudaModule_LIBRARIES}
#  ${VolumeRenderingCudaModule_CudaLibrary}
  )


# --------------------------------------------------------------------------
# Testing (requires some of the examples)

IF(BUILD_TESTING)
  SUBDIRS(Testing)
ENDIF(BUILD_TESTING)

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
ENDIF(CUDA_SUPPORT_ENABLED)
