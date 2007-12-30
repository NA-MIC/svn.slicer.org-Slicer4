PROJECT(CudaSupportModule)


INCLUDE(${CMAKE_SOURCE_DIR}/CMake/cuda/FindCuda.cmake)

# --------#
# Sources #
#---------#
 SET(CudaSupportModule_SRC
  ### BASIC CUDA RUNTIME API  CLASSES ###
    vtkCudaSupport.cxx
    vtkCudaDevice.cxx
    vtkCudaEvent.cxx
    vtkCudaBase.cxx
    vtkCudaStream.cxx
    vtkCudaOpenGLBufferObject.cxx

  #### CUDA MEMORY RUNTIME CLASSES ####
    vtkCudaMemoryBase.cxx
    vtkCudaMemory.cxx
    vtkCudaMemoryPitch.cxx
    vtkCudaHostMemory.cxx
    vtkCudaMemoryArray.cxx
    vtkCudaMemoryCopy.cxx

  #### CUDA Pipeline Classes ####
    vtkCudaImageDataFilter.cxx
    vtkCudaImageData.cxx

)

INCLUDE_DIRECTORIES(
  ${FOUND_CUT_INCLUDE}
  ${CUDA_INSTALL_PREFIX}/include
)

CUDA_INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR} 
                          ${FOUND_CUT_INCLUDE}
)

# --------------------------------------------------------------------------
# Wrapping

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE("${VTK_CMAKE_DIR}/vtkWrapTcl.cmake")
VTK_WRAP_TCL3(VolumeRenderingCudaModule
         CudaSupportModule_TCL_SRCS 
         "${CudaSupportModule_SRCS}" 
         "")

# --------------------------------------------------------------------------
# Build the library

## Set the Libraries we link
SET(VolumeRenderingCudaModule_LIBRARIES
  SlicerBaseLogic 
  SlicerBaseGUI 
  MRML
  vtkCommonTCL
  vtkImagingTCL
  vtkFilteringTCL
  
  ${CUDA_TARGET_LINK}
 )

ADD_LIBRARY(CudaSupportModule
  ${CudaSupportModule_SRCS}
  ${CudaSupportModule_TCL_SRCS}
  )

IF(BUILD_SHARED_LIBS)
  INSTALL(TARGETS CudaSupportModule
      RUNTIME DESTINATION bin COMPONENT RuntimeLibraries 
      LIBRARY DESTINATION lib COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION lib COMPONENT Development)
ENDIF(BUILD_SHARED_LIBS)

TARGET_LINK_LIBRARIES(CudaSupportModule
  ${CudaSupportModule_LIBRARIES}
  ${CudaSupportModule_CudaLibrary}
  )


# --------------------------------------------------------------------------
# Testing (requires some of the examples)

#IF(BUILD_TESTING)
#  SUBDIRS(Testing)
#ENDIF(BUILD_TESTING)


CONFIGURE_FILE(
  ${CudaSupportModule_SOURCE_DIR}/vtkCudaSupportModuleConfigure.h.in 
  ${CudaSupportModule_BINARY_DIR}/vtkCudaSupportModuleConfigure.h
)
CONFIGURE_FILE(
  ${CudaSupportModule_SOURCE_DIR}/presets.xml
  ${CudaSupportModule_BINARY_DIR}/presets.xml
  COPYONLY
)
