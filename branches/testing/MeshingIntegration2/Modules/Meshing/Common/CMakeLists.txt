PROJECT(MIMXCommon)


# Include dirs
#
INCLUDE_DIRECTORIES(
  ${Slicer3_INCLUDE_DIRS}
  ${SlicerBase_SOURCE_DIR}
  ${SlicerBase_BINARY_DIR}
  ${SlicerBaseLogic_SOURCE_DIR}
  ${SlicerBaseLogic_BINARY_DIR}
  ${SlicerBaseGUI_SOURCE_DIR}
  ${SlicerBaseGUI_BINARY_DIR}
  ${vtkITK_SOURCE_DIR}
  ${vtkITK_BINARY_DIR}
  ${ITK_INCLUDE_DIR}
  ${VTK_INCLUDE_DIR}
  ${mimxCommon_SOURCE_DIR} ${mimxCommon_BINARY_DIR} 
  ${MIMXCommon_SOURCE_DIR}../Adapter ${MIMXCommon_BINARY_DIR}../Adapter 
  ${MIMXAdapter_SOURCE_DIR} ${MIMXAdapter_BINARY_DIR} 
)

SET(LIB_NAME "mimxCommon")


SET ( LIB_SRCS 
    #mimxRegisterVersorRigidFilter.cxx  
    #mimxRegisterBSplineFilter.cxx 
    #mimxTransformIO.cxx 
    mimxCheckTypeOfFile.cxx
    # container types
    vtkContainer.cxx
    vtkLinkedListWrapper.cxx
    vtkLinkedListWrapperTree.cxx
    vtkMRMLFESurfaceNode.cxx
    vtkFESurfaceList.cxx
    vtkMimxCurve.cxx
    vtkMimxSTLSource.cxx
    vtkMimxUGrid.cxx
    vtkMimxBoundingBoxSource.cxx 
    vtkLocalAxesActor.cxx
    vtkPVAxesActor.cxx  
    vtkMimxSolidStructuredGridSource.cxx
    vtkMimxPlanarStructuredGridSource.cxx
    #vtkMeshQualityClass.cxx
    vtkMimxApplyNodeElementNumbers.cxx
    vtkMimxCreateIntervertebralDiscElementSets.cxx
    vtkMimxMergeNodesPointAndCellData.cxx
    vtkMimxMeshSeed.cxx
    vtkMimxActorBase.cxx
    vtkMimxMeshActor.cxx
    vtkMimxSurfacePolyDataActor.cxx
    vtkMimxUnstructuredGridActor.cxx
    vtkMimxErrorCallback.cxx
    vtkMimxImageActor.cxx
#    vtkMimxInteractorStyleUser.cxx 
    vtkMimxColorCodeMeshSeedActor.cxx
)

SET_SOURCE_FILES_PROPERTIES(
  mimxCheckTypeOfFile.cxx
  vtkMimxActorBase.cxx
  vtkMimxMeshActor.cxx
  vtkMimxSTLSource.cxx
  vtkMimxUGrid.cxx
  vtkMimxBoundingBoxSource.cxx
  vtkMimxErrorCallback.cxx
  vtkMimxImageActor.cxx
  vtkMimxMeshSeed.cxx
  vtkMimxMergeNodesPointAndCellData.cxx
  vtkMimxCreateIntervertebralDiscElementSets.cxx
  vtkMimxApplyNodeElementNumbers.cxx
  vtkContainer.cxx
  vtkLinkedListWrapper.cxx
  vtkLinkedListWrapperTree.cxx
  vtkFESurfaceList.cxx
  vtkMimxSurfacePolyDataActor.cxx
  vtkMimxUnstructuredGridActor.cxx
  vtkMimxCurve.cxx
  WRAP_EXCLUDE
  )

SET_SOURCE_FILES_PROPERTIES(
  vtkMimxActorBase.cxx
  ABSTRACT
  )


#
# Wrapping
#

#INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
#INCLUDE("${KWWidgets_CMAKE_DIR}/KWWidgetsWrappingMacros.cmake")
#KWWidgets_WRAP_TCL(${LIB_NAME} LIB_TCL_SRCS "${LIB_SRCS}" "")

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE("${VTK_CMAKE_DIR}/vtkWrapTcl.cmake")
VTK_WRAP_TCL3(${LIB_NAME} LIB_TCL_SRCS "${LIB_SRCS}" "")


ADD_LIBRARY( ${LIB_NAME} ${LIB_TCL_SRCS} ${LIB_SRCS} )
#ADD_LIBRARY( ${LIB_NAME}  ${LIB_SRCS} )

TARGET_LINK_LIBRARIES(${LIB_NAME}
  vtkCommonTCL
  vtkImagingTCL
  vtkFilteringTCL
  vtkIOTCL
  vtkRendering
  ITKCommon
  ITKBasicFilters
  ITKIO
  ITKAlgorithms
  ${KWWidgets_LIBRARIES}
  MRML
  )





