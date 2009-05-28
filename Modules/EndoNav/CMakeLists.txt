PROJECT(EndoNav)

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


# --------------------------------------------------------------------------
# Include dirs

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${Slicer3_Libs_INCLUDE_DIRS}
  ${Slicer3_Base_INCLUDE_DIRS}
  ${OpenIGTLink_INCLUDE_DIRS}
  )

#
# Configure include file
#
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/vtkEndoNavConfigure.h.in 
  ${CMAKE_CURRENT_BINARY_DIR}/vtkEndoNavConfigure.h
  )

file(GLOB headers 
  "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
  )

install(FILES 
  ${headers} 
  "${CMAKE_CURRENT_BINARY_DIR}/vtkEndoNavConfigure.h"
  DESTINATION ${Slicer3_INSTALL_MODULES_INCLUDE_DIR}/${PROJECT_NAME} COMPONENT Development
  )


# --------------------------------------------------------------------------
# Sources

set(EndoNav_SRCS 
  vtkEndoNavLogic.cxx
  vtkEndoNavGUI.cxx
  vtkIGTLConnector.cxx
  vtkIGTLCircularBuffer.cxx
  vtkIGTLToMRMLBase.cxx
  vtkIGTLToMRMLLinearTransform.cxx
  vtkIGTLToMRMLImage.cxx
  vtkIGTLToMRMLPosition.cxx
  vtkIGTLToViewerTransform.cxx
  vtkIGTLToViewerImage.cxx
)

# --------------------------------------------------------------------------
# Wrapping

include("${VTK_CMAKE_DIR}/vtkWrapTcl.cmake")
vtk_wrap_tcl3(EndoNav EndoNav_TCL_SRCS "${EndoNav_SRCS}" "")



#---------------------------------------------------------------------------
# Add Loadable Module support

generatelm(EndoNav_SRCS EndoNav.txt)

# --------------------------------------------------------------------------
# Build and install the library

set(lib_name EndoNav)

add_library(${lib_name}
  ${EndoNav_SRCS} 
  ${EndoNav_TCL_SRCS}
  )
slicer3_set_modules_output_path(${lib_name})


target_link_libraries(${lib_name}
  ${Slicer3_Libs_LIBRARIES}
  ${Slicer3_Base_LIBRARIES}
  ${KWWidgets_LIBRARIES}
  ${ITK_LIBRARIES}
  ${OpenIGTLink_LIBRARIES}
  )
slicer3_install_modules(${lib_name})

