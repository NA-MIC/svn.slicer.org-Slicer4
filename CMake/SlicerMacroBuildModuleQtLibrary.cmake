################################################################################
#
#  Program: 3D Slicer
#
#  Copyright (c) 2010 Kitware Inc.
#
#  See Doc/copyright/copyright.txt
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
# SlicerMacroBuildModuleQtLibrary
#

MACRO(SlicerMacroBuildModuleQtLibrary)
  SLICER_PARSE_ARGUMENTS(MODULEQTLIBRARY
    "NAME;EXPORT_DIRECTIVE;SRCS;MOC_SRCS;UI_SRCS;INCLUDE_DIRECTORIES;TARGET_LIBRARIES;RESOURCES"
    ""
    ${ARGN}
    )

  # --------------------------------------------------------------------------
  # Sanity checks
  # --------------------------------------------------------------------------
  SET(expected_defined_vars NAME EXPORT_DIRECTIVE)
  FOREACH(var ${expected_defined_vars})
    IF(NOT DEFINED MODULEQTLIBRARY_${var})
      MESSAGE(FATAL_ERROR "${var} is mandatory")
    ENDIF()
  ENDFOREACH()

  # --------------------------------------------------------------------------
  # Define library name
  # --------------------------------------------------------------------------
  SET(lib_name ${MODULEQTLIBRARY_NAME})

  # --------------------------------------------------------------------------
  # Include dirs
  # --------------------------------------------------------------------------
  INCLUDE_DIRECTORIES(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${MODULEQTLIBRARY_INCLUDE_DIRECTORIES}
    )

  #-----------------------------------------------------------------------------
  # Configure export header
  #-----------------------------------------------------------------------------
  SET(MY_LIBRARY_EXPORT_DIRECTIVE ${MODULEQTLIBRARY_EXPORT_DIRECTIVE})
  SET(MY_EXPORT_HEADER_PREFIX ${MODULEQTLIBRARY_NAME})
  SET(MY_LIBNAME ${lib_name})

  # Sanity checks
  IF (NOT EXISTS ${Slicer_EXPORT_HEADER_TEMPLATE})
    MESSAGE(FATAL_ERROR "error: Slicer_EXPORT_HEADER_TEMPLATE doesn't exist: ${Slicer_EXPORT_HEADER_TEMPLATE}")
  ENDIF()

  CONFIGURE_FILE(
    ${Slicer_EXPORT_HEADER_TEMPLATE}
    ${CMAKE_CURRENT_BINARY_DIR}/${MY_EXPORT_HEADER_PREFIX}Export.h
    )
  SET(dynamicHeaders
    "${dynamicHeaders};${CMAKE_CURRENT_BINARY_DIR}/${MY_EXPORT_HEADER_PREFIX}Export.h")

  #-----------------------------------------------------------------------------
  # Sources
  #-----------------------------------------------------------------------------
  QT4_WRAP_CPP(MODULEQTLIBRARY_MOC_OUTPUT ${MODULEQTLIBRARY_MOC_SRCS})
  QT4_WRAP_UI(MODULEQTLIBRARY_UI_CXX ${MODULEQTLIBRARY_UI_SRCS})
  IF(DEFINED MODULEQTLIBRARY_RESOURCES)
    QT4_ADD_RESOURCES(MODULEQTLIBRARY_QRC_SRCS ${MODULEQTLIBRARY_RESOURCES})
  ENDIF()

  IF (NOT EXISTS ${Slicer_LOGOS_RESOURCE})
    MESSAGE("Warning, Slicer_LOGOS_RESOURCE doesn't exist: ${Slicer_LOGOS_RESOURCE}")
  ENDIF()
  QT4_ADD_RESOURCES(MODULEQTLIBRARY_QRC_SRCS ${Slicer_LOGOS_RESOURCE})

  SET_SOURCE_FILES_PROPERTIES(
    ${MODULEQTLIBRARY_UI_CXX}
    ${MODULEQTLIBRARY_SRCS}
    WRAP_EXCLUDE
    )

  # --------------------------------------------------------------------------
  # Source groups
  # --------------------------------------------------------------------------
  SOURCE_GROUP("Resources" FILES
    ${MODULEQTLIBRARY_UI_SRCS}
    ${Slicer_LOGOS_RESOURCE}
    ${MODULEQTLIBRARY_RESOURCES}
    )

  SOURCE_GROUP("Generated" FILES
    ${MODULEQTLIBRARY_UI_CXX}
    ${MODULEQTLIBRARY_MOC_OUTPUT}
    ${MODULEQTLIBRARY_QRC_SRCS}
    ${dynamicHeaders}
    )

  # --------------------------------------------------------------------------
  # Build library
  #-----------------------------------------------------------------------------
  ADD_LIBRARY(${lib_name}
    ${MODULEQTLIBRARY_SRCS}
    ${MODULEQTLIBRARY_MOC_OUTPUT}
    ${MODULEQTLIBRARY_UI_CXX}
    ${MODULEQTLIBRARY_QRC_SRCS}
    )
  
  # Set qt loadable modules output path
  SET_TARGET_PROPERTIES(${lib_name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${Slicer_QTLOADABLEMODULES_BIN_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${Slicer_QTLOADABLEMODULES_LIB_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${Slicer_QTLOADABLEMODULES_LIB_DIR}"
    )
  SET_TARGET_PROPERTIES(${lib_name} PROPERTIES LABELS ${lib_name})

  TARGET_LINK_LIBRARIES(${lib_name}
    ${MODULEQTLIBRARY_TARGET_LIBRARIES}
    )

  # Apply user-defined properties to the library target.
  IF(Slicer_LIBRARY_PROPERTIES)
    SET_TARGET_PROPERTIES(${lib_name} PROPERTIES ${Slicer_LIBRARY_PROPERTIES})
  ENDIF()

  # --------------------------------------------------------------------------
  # Install library
  # --------------------------------------------------------------------------
  INSTALL(TARGETS ${lib_name}
    RUNTIME DESTINATION ${Slicer_INSTALL_QTLOADABLEMODULES_BIN_DIR} COMPONENT RuntimeLibraries
    LIBRARY DESTINATION ${Slicer_INSTALL_QTLOADABLEMODULES_LIB_DIR} COMPONENT RuntimeLibraries
    ARCHIVE DESTINATION ${Slicer_INSTALL_QTLOADABLEMODULES_LIB_DIR} COMPONENT Development
    )

  # --------------------------------------------------------------------------
  # Install headers
  # --------------------------------------------------------------------------
  IF(DEFINED Slicer_DEVELOPMENT_INSTALL)
    IF(NOT DEFINED ${MODULEQTLIBRARY_NAME}_DEVELOPMENT_INSTALL)
      SET(${MODULEQTLIBRARY_NAME}_DEVELOPMENT_INSTALL ${Slicer_DEVELOPMENT_INSTALL})
    ENDIF()
  ELSE()
    IF (NOT DEFINED ${MODULEQTLIBRARY_NAME}_DEVELOPMENT_INSTALL)
      SET(${MODULEQTLIBRARY_NAME}_DEVELOPMENT_INSTALL OFF)
    ENDIF()
  ENDIF()

  IF(${MODULEQTLIBRARY_NAME}_DEVELOPMENT_INSTALL)
    # Install headers
    FILE(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
    INSTALL(FILES
      ${headers}
      ${dynamicHeaders}
      DESTINATION ${Slicer_INSTALL_QTLOADABLEMODULES_INCLUDE_DIR}/${MODULEQTLIBRARY_NAME} COMPONENT Development
      )
  ENDIF()

  # --------------------------------------------------------------------------
  # Export target
  # --------------------------------------------------------------------------
  SET_PROPERTY(GLOBAL APPEND PROPERTY Slicer_TARGETS ${MODULEQTLIBRARY_NAME})
ENDMACRO()
