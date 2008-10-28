PROJECT(MeshingWorkflow)

cmake_minimum_required(VERSION 2.4)
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

# --------------------------------------------------------------------------
# Find Slicer3

if(NOT Slicer3_SOURCE_DIR)
  find_package(Slicer3 REQUIRED)
  include(${Slicer3_USE_FILE})
  slicer3_set_default_install_prefix_for_external_projects()
endif(NOT Slicer3_SOURCE_DIR)


#SET(VERDICT_DIR ".")
#SET(VERDICT_SRCS  
#        "${VERDICT_DIR}/V_HexMetric.cpp" 
#        "${VERDICT_DIR}/V_GaussIntegration.cpp"
#        "${VERDICT_DIR}/V_TetMetric.cpp" 
#        "${VERDICT_DIR}/VerdictVector.cpp" 
#       )

#   SET(VERDICT_HEADERS  
#        "${VERDICT_DIR}/verdict_defines.hpp" 
#        "${VERDICT_DIR}/verdict.h" 
#        "${VERDICT_DIR}/VerdictVector.hpp" 
#        "${VERDICT_DIR}/V_GaussIntegration.hpp" 
#        "${VERDICT_DIR}/v_vector.h" 
#        )

# --------------------------------------------------------------------------
# Include dirs
#
INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${Slicer3_Libs_INCLUDE_DIRS}
  ${Slicer3_Base_INCLUDE_DIRS}
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
 ) 
 
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/vtkMeshingWorkflowConfigure.h.in 
  ${CMAKE_CURRENT_BINARY_DIR}/vtkMeshingWorkflowConfigure.h
  )

file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES 
  ${headers} 
  "${CMAKE_CURRENT_BINARY_DIR}/vtkMeshingWorkflowConfigure.h"
  DESTINATION ${Slicer3_INSTALL_MODULES_INCLUDE_DIR}/${PROJECT_NAME} COMPONENT Development
  )
  
 # until slicer goes to vtk5.2, we need to specially define a portion of the Verdict library to have 
 # advanced mesh rendering functions 
  

SET(LIB_NAME MeshingWorkflow)
SET(MeshingWorkflow_SRCS 
  vtkMeshingWorkflowGUI.cxx
  vtkMeshingWorkflowLogic.cxx
  vtkMRMLMeshingWorkflowNode.cxx
  #vtkMeshingWorkflowMRMLNotebook.cxx
  #vtkMeshQualityExtended.cxx
  #${VERDICT_SOURCES}
  )
   

#INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
#INCLUDE("${VTK_CMAKE_DIR}/vtkWrapTcl.cmake")
#VTK_WRAP_TCL3(${LIB_NAME} LIB_TCL_SRCS "${LIB_SRCS}" "")

include("${VTK_CMAKE_DIR}/vtkWrapTcl.cmake")
vtk_wrap_tcl3(MeshingWorkflow 
   MeshingWorkflow_TCL_SRCS 
  "${MeshingWorkflow_SRCS}" "")
  
  
#---------------------------------------------------------------------------
# Add Loadable Module support

generatelm(MeshingWorkflow_SRCS 
 MeshingWorkflow.txt
 )
  

# Create the library. The library is built in static mode for convenience. 
# Check the 'Callbacks' example for more information about building it in
# shared mode, i.e. without the STATIC keyword (Win32 compilers requires
# an additional header file to setup DLL export symbols correctly).

ADD_LIBRARY(${LIB_NAME} ${MeshingWorkflow_TCL_SRCS} ${MeshingWorkflow_SRCS})

slicer3_set_modules_output_path(${LIB_NAME})

TARGET_LINK_LIBRARIES(${LIB_NAME}
  ${Slicer3_Libs_LIBRARIES}
  ${Slicer3_Base_LIBRARIES}
  ${KWWidgets_LIBRARIES} 
  mimxWidgets
  mimxMeshIO
  mimxWidgets
  mimxFilter
  mimxCommon
  BuildingBlock
  )

slicer3_install_modules(${LIB_NAME})
