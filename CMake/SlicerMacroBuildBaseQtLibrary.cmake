################################################################################
#
#  Program: 3D Slicer
#
#  Copyright (c) Kitware Inc.
#
#  See COPYRIGHT.txt
#  or http://www.slicer.org/copyright/copyright.txt for details.
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc.
#  and was partially funded by NIH grant 3P41RR013218-12S1
#
################################################################################

#
# SlicerMacroBuildBaseQtLibrary
#

#
# Parameters:
#
#   NAME .................: Name of the library
#
#   EXPORT_DIRECTIVE .....: Export directive that should be used to export symbol
#
#   SRCS .................: List of source files
#
#   MOC_SRCS .............: Optional list of headers to run through the meta object compiler (moc)
#                           using QT4_WRAP_CPP CMake macro
#
#   UI_SRCS ..............: Optional list of UI file to run through UI compiler (uic) using
#                           QT4_WRAP_UI CMake macro
#
#   INCLUDE_DIRECTORIES ..: Optional list of extra folder that should be included. See implementation
#                           for the list of folder included by default.
#
#   TARGET_LIBRARIES .....: Optional list of target libraries that should be used with TARGET_LINK_LIBRARIES
#                           CMake macro. See implementation for the list of libraries added by default.
#
#   RESOURCES ............: Optional list of files that should be converted into resource header
#                           using QT4_ADD_RESOURCES
#
# Options:
#
#   WRAP_PYTHONQT ........: If specified, the sources (SRCS) will be 'PythonQt' wrapped and a static
#                           library named <NAME>PythonQt will be built.
#

macro(SlicerMacroBuildBaseQtLibrary)
  SLICER_PARSE_ARGUMENTS(SLICERQTBASELIB
    "NAME;EXPORT_DIRECTIVE;SRCS;MOC_SRCS;UI_SRCS;INCLUDE_DIRECTORIES;TARGET_LIBRARIES;RESOURCES"
    "WRAP_PYTHONQT"
    ${ARGN}
    )
  message(STATUS "Configuring Slicer Qt base library: ${SLICERQTBASELIB_NAME}")
  # --------------------------------------------------------------------------
  # Sanity checks
  # --------------------------------------------------------------------------
  set(expected_defined_vars NAME EXPORT_DIRECTIVE)
  foreach(var ${expected_defined_vars})
    if(NOT DEFINED SLICERQTBASELIB_${var})
      message(FATAL_ERROR "${var} is mandatory")
    endif()
  endforeach()

  if(NOT DEFINED Slicer_INSTALL_NO_DEVELOPMENT)
    message(SEND_ERROR "Slicer_INSTALL_NO_DEVELOPMENT is mandatory")
  endif()

  # --------------------------------------------------------------------------
  # Define library name
  # --------------------------------------------------------------------------
  set(lib_name ${SLICERQTBASELIB_NAME})

  # --------------------------------------------------------------------------
  # Include dirs
  # --------------------------------------------------------------------------

  set(include_dirs
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${Slicer_Base_INCLUDE_DIRS}
    ${Slicer_Libs_INCLUDE_DIRS}
    ${SLICERQTBASELIB_INCLUDE_DIRECTORIES}
    )

  include_directories(${include_dirs})

  #-----------------------------------------------------------------------------
  # Update Slicer_Base_INCLUDE_DIRS
  #-----------------------------------------------------------------------------
  set(Slicer_Base_INCLUDE_DIRS ${Slicer_Base_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    CACHE INTERNAL "Slicer Base includes" FORCE)

  #-----------------------------------------------------------------------------
  # Configure
  # --------------------------------------------------------------------------
  set(MY_LIBRARY_EXPORT_DIRECTIVE ${SLICERQTBASELIB_EXPORT_DIRECTIVE})
  set(MY_EXPORT_HEADER_PREFIX ${SLICERQTBASELIB_NAME})
  set(MY_LIBNAME ${lib_name})

  configure_file(
    ${Slicer_SOURCE_DIR}/CMake/qSlicerExport.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/${MY_EXPORT_HEADER_PREFIX}Export.h
    )
  set(dynamicHeaders
    "${dynamicHeaders};${CMAKE_CURRENT_BINARY_DIR}/${MY_EXPORT_HEADER_PREFIX}Export.h")

  #-----------------------------------------------------------------------------
  # Sources
  # --------------------------------------------------------------------------
  QT4_WRAP_CPP(SLICERQTBASELIB_MOC_OUTPUT ${SLICERQTBASELIB_MOC_SRCS})
  QT4_WRAP_UI(SLICERQTBASELIB_UI_CXX ${SLICERQTBASELIB_UI_SRCS})
  if(DEFINED SLICERQTBASELIB_RESOURCES)
    QT4_ADD_RESOURCES(SLICERQTBASELIB_QRC_SRCS ${SLICERQTBASELIB_RESOURCES})
  endif(DEFINED SLICERQTBASELIB_RESOURCES)

  QT4_ADD_RESOURCES(SLICERQTBASELIB_QRC_SRCS ${Slicer_SOURCE_DIR}/Resources/qSlicer.qrc)

  set_source_files_properties(
    ${SLICERQTBASELIB_UI_CXX}
    ${SLICERQTBASELIB_MOC_OUTPUT}
    ${SLICERQTBASELIB_QRC_SRCS}
    WRAP_EXCLUDE
    )

  # --------------------------------------------------------------------------
  # Source groups
  # --------------------------------------------------------------------------
  source_group("Resources" FILES
    ${SLICERQTBASELIB_UI_SRCS}
    ${Slicer_SOURCE_DIR}/Resources/qSlicer.qrc
    ${SLICERQTBASELIB_RESOURCES}
  )

  source_group("Generated" FILES
    ${SLICERQTBASELIB_UI_CXX}
    ${SLICERQTBASELIB_MOC_OUTPUT}
    ${SLICERQTBASELIB_QRC_SRCS}
    ${dynamicHeaders}
  )

  # --------------------------------------------------------------------------
  # Build the library
  # --------------------------------------------------------------------------
  add_library(${lib_name}
    ${SLICERQTBASELIB_SRCS}
    ${SLICERQTBASELIB_MOC_OUTPUT}
    ${SLICERQTBASELIB_UI_CXX}
    ${SLICERQTBASELIB_QRC_SRCS}
    )
  set_target_properties(${lib_name} PROPERTIES LABELS ${lib_name})

  # Apply user-defined properties to the library target.
  if(Slicer_LIBRARY_PROPERTIES)
    set_target_properties(${lib_name} PROPERTIES ${Slicer_LIBRARY_PROPERTIES})
  endif()

  target_link_libraries(${lib_name}
    ${QT_LIBRARIES}
    ${CTK_EXTERNAL_LIBRARIES}
    ${SLICERQTBASELIB_TARGET_LIBRARIES}
    )

  #-----------------------------------------------------------------------------
  # Install library
  #-----------------------------------------------------------------------------
  install(TARGETS ${lib_name}
    RUNTIME DESTINATION ${Slicer_INSTALL_BIN_DIR} COMPONENT RuntimeLibraries
    LIBRARY DESTINATION ${Slicer_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
    ARCHIVE DESTINATION ${Slicer_INSTALL_LIB_DIR} COMPONENT Development
  )

  # --------------------------------------------------------------------------
  # Install headers
  # --------------------------------------------------------------------------
  if(NOT Slicer_INSTALL_NO_DEVELOPMENT)
    # Install headers
    file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
    install(FILES
      ${headers}
      ${dynamicHeaders}
      DESTINATION ${Slicer_INSTALL_INCLUDE_DIR}/${PROJECT_NAME} COMPONENT Development
      )
  endif()

  # --------------------------------------------------------------------------
  # PythonQt wrapping
  # --------------------------------------------------------------------------
  if(Slicer_USE_PYTHONQT AND SLICERQTBASELIB_WRAP_PYTHONQT)
    ctkMacroBuildLibWrapper(
      NAMESPACE "org.slicer.base"
      TARGET ${lib_name}
      SRCS "${SLICERQTBASELIB_SRCS}"
      INSTALL_BIN_DIR ${Slicer_INSTALL_BIN_DIR}
      INSTALL_LIB_DIR ${Slicer_INSTALL_LIB_DIR}
      )
  endif()

  # --------------------------------------------------------------------------
  # Export target
  # --------------------------------------------------------------------------
  set_property(GLOBAL APPEND PROPERTY Slicer_TARGETS ${SLICERQTBASELIB_NAME})

endmacro()
