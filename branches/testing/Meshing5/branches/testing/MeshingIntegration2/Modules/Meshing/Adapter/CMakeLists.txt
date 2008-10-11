PROJECT(MIMXAdapter)

SET (MIMXAdapter_SRCS 
    vtkUnstructuredGridToitkMesh.cxx 
    itkMeshTovtkPolyData.cxx  
    vtkPolyDataToitkMesh.cxx
    ) 

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
  ${ITK_INCLUDE_DIR}
  ${VTK_INCLUDE_DIR}
  ${MIMXCommon_SOURCE_DIR} ${MIMXCommon_BINARY_DIR} 
)

#
# Wrapping
#
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR} ${MIMXAdapter_BINARY_DIR})
INCLUDE("${VTK_CMAKE_DIR}/vtkWrapTcl.cmake")
VTK_WRAP_TCL3(mimxAdapter MimxAdapter_TCL_SRCS "${MIMXAdapter_SRCS}" "")

ADD_LIBRARY(mimxAdapter ${MIMXAdapter_SRCS} )

TARGET_LINK_LIBRARIES(mimxAdapter 
  vtkCommonTCL
  vtkImagingTCL
  vtkFilteringTCL
  vtkIOTCL
  vtkWidgets
  vtkRendering
  ITKCommon
  ITKBasicFilters
  ITKIO
  ITKAlgorithms
  )


IF(BUILD_SHARED_LIBS)
  INSTALL_TARGETS(${SLICER3_LIB_INSTALL_DIR} mimxAdapter)
ENDIF(BUILD_SHARED_LIBS)



