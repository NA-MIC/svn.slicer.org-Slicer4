PROJECT(MeshingWorkflow)

# directories for vtkMIMX
INCLUDE_DIRECTORIES( ${BOUNDINGBOX_SOURCE_DIR}/../Common ${BOUNDINGBOX_BINARY_DIR}/../Common ) 
INCLUDE_DIRECTORIES( ${BOUNDINGBOX_SOURCE_DIR}/../Filter ${BOUNDINGBOX_BINARY_DIR}/../Filter ) 
INCLUDE_DIRECTORIES( ${BOUNDINGBOX_SOURCE_DIR}/../Widgets ${BOUNDINGBOX_BINARY_DIR}/../Widgets ) 
INCLUDE_DIRECTORIES( ${BOUNDINGBOX_SOURCE_DIR}/../BoundingBox ${BOUNDINGBOX_BINARY_DIR}/../BoundingBox ) 

# --------------------------------------------------------------------------
# Include dirs
# MESSAGE(STATUS "QDecModle: Qdec binary dir = ${Qdec_BINARY_DIR}")
INCLUDE_DIRECTORIES(
  ${ModuleDescriptionParser_SOURCE_DIR}
  ${ModuleDescriptionParser_BINARY_DIR}
  ${MeshingWorkflow_SOURCE_DIR}
  ${MeshingWorkflow_BINARY_DIR}
  ${BOUNDINGBOX_SOURCE_DIR}
  ${BOUNDINGBOX_BINARY_DIR}
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

SET(LIB_NAME "MeshingWorkflow")
SET(LIB_SRCS 
  vtkMeshingWorkflowGUI.cxx
  vtkMeshingWorkflowLogic.cxx
  vtkMRMLMeshingWorkflowNode.cxx
  vtkMRMLFESurfaceNode.cxx
  vtkMRMLFiniteElementBoundingBoxNode.cxx
  vtkMRMLFiniteElementMeshNode.cxx
  vtkFESurfaceList.cxx
  vtkFiniteElementBoundingBoxList.cxx
  vtkFiniteElementMeshList.cxx
  vtkMeshingWorkflowMRMLNotebook.cxx
  vtkFESurfaceMRMLMenuGroup.cxx
  vtkFiniteElementMRMLMeshMenuGroup.cxx
  vtkFiniteElementMRMLCreateBBFromBoundsGroup.cxx
  vtkFiniteElementCreateBBMeshSeedGroup.cxx
  vtkFiniteElementEditBBGroup.cxx
  vtkFiniteElementCreateFEMeshFromBBGroup.cxx
  vtkMRMLFiniteElementBoundingBoxDisplayNode.cxx
    vtkMRMLFiniteElementMeshDisplayNode.cxx
  )
  


#INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
#INCLUDE("${KWWidgets_CMAKE_DIR}/KWWidgetsWrappingMacros.cmake")
#KWWidgets_WRAP_TCL(${LIB_NAME} LIB_TCL_SRCS "${LIB_SRCS}" "")

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE("${VTK_CMAKE_DIR}/vtkWrapTcl.cmake")
VTK_WRAP_TCL3(${LIB_NAME} LIB_TCL_SRCS "${LIB_SRCS}" "")



# Create the library. The library is built in static mode for convenience. 
# Check the 'Callbacks' example for more information about building it in
# shared mode, i.e. without the STATIC keyword (Win32 compilers requires
# an additional header file to setup DLL export symbols correctly).

ADD_LIBRARY(${LIB_NAME} ${LIB_TCL_SRCS} ${LIB_SRCS})

IF(BUILD_SHARED_LIBS)
  INSTALL_TARGETS(${SLICER3_LIB_INSTALL_DIR} ${LIB_NAME})
ENDIF(BUILD_SHARED_LIBS)

TARGET_LINK_LIBRARIES(${LIB_NAME}
  ModuleDescriptionParser
  SlicerBaseLogic 
  SlicerBaseGUI 
  MRML
  vtkCommonTCL
  vtkImagingTCL
  vtkFilteringTCL
  vtkIOTCL
  vtkWidgets
  ${KWWidgets_LIBRARIES} 
  mimxWidgets
  mimxFilter
  mimxCommon
  mimxBoundingBox
  )

CONFIGURE_FILE(
  ${MeshingWorkflow_SOURCE_DIR}/vtkMeshingWorkflowConfigure.h.in 
  ${MeshingWorkflow_BINARY_DIR}/vtkMeshingWorkflowConfigure.h
)

