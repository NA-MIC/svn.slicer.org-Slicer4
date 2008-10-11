PROJECT(MeshingWorkflow)

SET(VERDICT_DIR ".")
SET(VERDICT_SRCS  
        "${VERDICT_DIR}/V_HexMetric.cpp" 
        "${VERDICT_DIR}/V_GaussIntegration.cpp"
        "${VERDICT_DIR}/V_TetMetric.cpp" 
        "${VERDICT_DIR}/VerdictVector.cpp" 
       )

   SET(VERDICT_HEADERS  
        "${VERDICT_DIR}/verdict_defines.hpp" 
        "${VERDICT_DIR}/verdict.h" 
        "${VERDICT_DIR}/VerdictVector.hpp" 
        "${VERDICT_DIR}/V_GaussIntegration.hpp" 
        "${VERDICT_DIR}/v_vector.h" 
        )

 
# directories for vtkMIMX
#INCLUDE_DIRECTORIES( ${BOUNDINGBOX_SOURCE_DIR}/../Common ${BOUNDINGBOX_BINARY_DIR}/../Common ) 
#INCLUDE_DIRECTORIES( ${BOUNDINGBOX_SOURCE_DIR}/../Filter ${BOUNDINGBOX_BINARY_DIR}/../Filter ) 
#INCLUDE_DIRECTORIES( ${BOUNDINGBOX_SOURCE_DIR}/../Widgets ${BOUNDINGBOX_BINARY_DIR}/../Widgets ) 
#INCLUDE_DIRECTORIES( ${BOUNDINGBOX_SOURCE_DIR}/../BoundingBox ${BOUNDINGBOX_BINARY_DIR}/../BoundingBox ) 

# --------------------------------------------------------------------------
# Include dirs
#
INCLUDE_DIRECTORIES(
  ${SlicerBase_SOURCE_DIR}
  ${SlicerBase_BINARY_DIR}
  ${SlicerBaseLogic_SOURCE_DIR}
  ${SlicerBaseLogic_BINARY_DIR}
  ${SlicerBaseGUI_SOURCE_DIR}
  ${SlicerBaseGUI_BINARY_DIR}
  ${vtkITK_SOURCE_DIR}
  ${vtkITK_BINARY_DIR}
  ${MRML_SOURCE_DIR} ${MRML_BINARY_DIR}
  ${ITK_INCLUDE_DIR}
  ${VTK_INCLUDE_DIR}
  ${MIMXCommon_SOURCE_DIR} 
  ${MIMXCommon_BINARY_DIR}
  ${MIMXAdapter_SOURCE_DIR} 
  ${MIMXAdapter_BINARY_DIR} 
  ${MIMXMesh_SOURCE_DIR} 
  ${MIMXMesh_BINARY_DIR} 
  ${MIMXWidgets_SOURCE_DIR} 
  ${MIMXWidgets_BINARY_DIR} 
  ${BUILDINGBLOCK_SOURCE_DIR} 
  ${BUILDINGBLOCK_BINARY_DIR} 
  /usr/local/slicer3trunk/Slicer3/Modules/Meshing/BuildingBlock
 ) 
  
 # until slicer goes to vtk5.2, we need to specially define a portion of the Verdict library to have 
 # advanced mesh rendering functions 
  

SET(LIB_NAME "MeshingWorkflow")
SET(LIB_SRCS 
  vtkMeshingWorkflowGUI.cxx
  vtkMeshingWorkflowLogic.cxx
  vtkMRMLMeshingWorkflowNode.cxx
  vtkMeshingWorkflowMRMLNotebook.cxx
#  vtkMeshQualityExtended.cxx
#  ${VERDICT_SOURCES}
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
  BuildingBlock
  )

CONFIGURE_FILE(
  ${MeshingWorkflow_SOURCE_DIR}/vtkMeshingWorkflowConfigure.h.in 
  ${MeshingWorkflow_BINARY_DIR}/vtkMeshingWorkflowConfigure.h
)

