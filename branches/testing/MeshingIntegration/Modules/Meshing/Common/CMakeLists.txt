PROJECT(MIMXCommon)

CONFIGURE_FILE(
  ${MIMXCommon_SOURCE_DIR}/vtkCommonConfigure.h.in 
  ${MIMXCommon_BINARY_DIR}/vtkCommonConfigure.h
)

SET ( COMMON_SRCS 
  #mimxRegisterVersorRigidFilter.cxx  
  #mimxRegisterBSplineFilter.cxx 
  #mimxTransformIO.cxx 
    mimxCheckTypeOfFile.cxx
    vtkContainer.cxx
    vtkLinkedListWrapper.cxx
    vtkMimxCurve.cxx
  #vtkMimxSTLSource.cxx
  #vtkMimxUGrid.cxx
    vtkMimxBoundingBoxSource.cxx 
    vtkLocalAxesActor.cxx
    vtkPVAxesActor.cxx  
    vtkMimxSolidStructuredGridSource.cxx
    vtkMimxPlanarStructuredGridSource.cxx
    vtkMeshQualityClass.cxx
    vtkMimxApplyNodeElementNumbers.cxx
    vtkMimxCreateIntervertebralDiscElementSets.cxx
    vtkMimxMergeNodesPointAndCellData.cxx
    vtkMimxMeshSeed.cxx
    vtkMimxActorBase.cxx
    vtkMimxSurfacePolyDataActor.cxx
    vtkMimxUnstructuredGridActor.cxx
)

INCLUDE_DIRECTORIES( ${MIMXCommon_SOURCE_DIR} ${MIMXCommon_BINARY_DIR} ) 

SET_SOURCE_FILES_PROPERTIES(
  mimxCheckTypeOfFile.cxx
  WRAP_EXCLUDE
  )

SET_SOURCE_FILES_PROPERTIES(
  vtkMimxActorBase.cxx
  ABSTRACT
  )

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE("${VTK_CMAKE_DIR}/vtkWrapTcl.cmake")
VTK_WRAP_TCL3(MIMXCommon COMMON_TCL_SRCS "${COMMON_SRCS}" "")

ADD_LIBRARY( mimxCommon ${COMMON_TCL_SRCS} ${COMMON_SRCS} )

TARGET_LINK_LIBRARIES(mimxCommon
  vtkCommonTCL
  vtkImagingTCL
  vtkFilteringTCL
  vtkIOTCL
  vtkWidgets
  vtkRendering
  ${KWWidgets_LIBRARIES} 
  )

IF(BUILD_SHARED_LIBS)
  INSTALL_TARGETS(${SLICER3_LIB_INSTALL_DIR} mimxCommon)
ENDIF(BUILD_SHARED_LIBS)

