PROJECT(RegistrationCudaModule)

## ONLY BUILD THIS IF YOU WANT CUDA SUPPORT TO BE ENABLED
IF(CUDA_SUPPORT_ENABLED)

OPTION(REGISTRATIONCUDAMODULE "Would you like to build Registration Module with CUDA support" OFF)
if(REGISTRATIONCUDAMODULE)

INCLUDE(${CMAKE_SOURCE_DIR}/CMake/cuda/FindCuda.cmake)

# --------------------------------------------------------------------------
# Sources
  SET(RegistrationCudaModule_SRCS
        ntk2DBSpline3Transform.cpp
        ntk2DData.cpp
        ntk2DData16.cpp
        ntk2DDataResizer.cpp
        ntk2DElasticRegistration.cpp
        ntk2DElasticTransformation.cpp
        ntk3DDSFFiltering.cpp
        ntk3DData.cpp
        ntk3DData16.cpp
        ntk3DDataCropper.cpp
        ntk3DDataResizer.cpp
        ntk4DData.cpp
        ntk4DData16.cpp
        ntkBSpline3Function.cpp
        ntkBSpline3Transform.cpp
        ntkCudaDeviceMemory.cpp
        ntkData.cpp
        ntkDataAligner.cpp
        ntkDeformationSpline.cpp
        ntkElasticRegistration.cpp
        ntkElasticRegistrationCUDA.cpp
        ntkElasticTransformation.cpp
        ntkElasticTransformationCUDA.cpp
        ntkElasticityMap.cpp
        ntkFiltering.cpp
        ntkGaussianFiltering.cpp
        ntkLinearTransformation.cpp
        ntkLinearTransformationCUDA.cpp
        ntkMISimilarityMeasure.cpp
        ntkMISimilarityMeasureCUDA.cpp
        ntkMatrix.cpp
        ntkMedianFiltering.cpp
        ntkMedianFilteringCUDA.cpp
        ntkProperties.cpp
        ntkRegistration.cpp
        ntkRigidRegistration.cpp
        ntkRigidRegistrationCUDA.cpp
        ntkSimilarityMeasure.cpp
        ntkTensor.cpp
        ntkTimer.cpp
        ntkTransformation.cpp
        Matrix.cxx
  )
  SET(Registration_ADDITIONAL_HEADERS
        CUDA_calculateMI.h
        CUDA_calculateSSD.h
        CUDA_calculateSSDGradient.h
        CUDA_elasticTransformation.h
        CUDA_linearTransformation.h
        ntk2DElasticRegistration.h
        ntk2DElasticTransformation.h
        ntkDeformationSpline.h
        ntkElasticRegistration.h
        ntkElasticRegistrationCUDA.h
        ntkElasticTransformation.h
        ntkElasticTransformationCUDA.h
        ntkElasticityMap.h
        ntkLinearTransformation.h
        ntkLinearTransformationCUDA.h
        ntkMISimilarityMeasure.h
        ntkMISimilarityMeasureCUDA.h
        ntkRegistration.h
        ntkRigidRegistration.h
        ntkRigidRegistrationCUDA.h
        ntkSimilarityMeasure.h
        ntkTransformation.h
        vtkRegistrationCudaModule.h
        vtkRegistrationCudaModuleWin32Header.h
      )
  
  SET(RegistrationCudaModule_CUDA_SRCS
        CUDA_calculateMI.cu
        CUDA_calculateSSD.cu
        CUDA_calculateSSDGradient.cu
        CUDA_elasticTransformation.cu
        CUDA_linearTransformation.cu
  )

#-----------------------------------------------
# Include dirs

INCLUDE_DIRECTORIES(
  ${FOUND_CUT_INCLUDE}
  ${CUDA_INSTALL_PREFIX}/include
  ${CudaSupport_SOURCE_DIR}
  ${CudaSupport_BINARY_DIR}
  ${RegistrationCudaModule_SOURCE_DIR}
  ${RegistrationCudaModule_BINARY_DIR}
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
                           ${VTK_INCLUDE_DIRS}
 )

# --------------------------------------------------------------------------
# Wrapping

#INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
#INCLUDE("${VTK_CMAKE_DIR}/vtkWrapTcl.cmake")
#VTK_WRAP_TCL3(RegistrationCudaModule
#         RegistrationCudaModule_TCL_SRCS 
#         "${RegistrationCudaModule_SRCS}" 
#         "")


# --------------------------------------------------------------------------
# Build the library

## Build the Cuda Library first
#CUDA_ADD_LIBRARY(RegistrationCudaModule_CudaLibrary  
#  )
  
#TARGET_LINK_LIBRARIES(RegistrationCudaModule_CudaLibrary
#  ${CudaSupport}
#)

## Set the Libraries we link
SET(RegistrationCudaModule_LIBRARIES
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
  CudaSupport
 )

CUDA_ADD_LIBRARY(RegistrationCudaModule
  ${RegistrationCudaModule_SRCS}
  ${RegistrationCudaModule_ADDITIONAL_HEADERS}
  ${RegistrationCudaModule_TCL_SRCS}
  ${RegistrationCudaModule_CUDA_SRCS}
  )

IF(BUILD_SHARED_LIBS)
  INSTALL(TARGETS RegistrationCudaModule
      RUNTIME DESTINATION bin COMPONENT RuntimeLibraries 
      LIBRARY DESTINATION lib COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION lib COMPONENT Development)
ENDIF(BUILD_SHARED_LIBS)

TARGET_LINK_LIBRARIES(RegistrationCudaModule
  ${RegistrationCudaModule_LIBRARIES}
#  ${RegistrationCudaModule_CudaLibrary}
  )


# --------------------------------------------------------------------------
# Testing (requires some of the examples)

##IF(BUILD_TESTING)
##  ADD_SUBDIRECTORY(Testing)
##ENDIF(BUILD_TESTING)

##IF(BUILD_EXAMPLES)
##  ADD_SUBDIRECTORY(Examples)
##ENDIF(BUILD_EXAMPLES)

CONFIGURE_FILE(
  ${RegistrationCudaModule_SOURCE_DIR}/vtkRegistrationCudaModuleConfigure.h.in 
  ${RegistrationCudaModule_BINARY_DIR}/vtkRegistrationCudaModuleConfigure.h
)
CONFIGURE_FILE(
  ${RegistrationCudaModule_SOURCE_DIR}/presets.xml
  ${RegistrationCudaModule_BINARY_DIR}/presets.xml
  COPYONLY
)
ENDIF(REGISTRATIONCUDAMODULE)
ENDIF(CUDA_SUPPORT_ENABLED)
