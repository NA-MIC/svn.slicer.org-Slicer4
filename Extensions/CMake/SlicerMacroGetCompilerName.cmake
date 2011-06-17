
#
# Adapted from CMake/Modules/CTest.cmake
#

MACRO(SlicerMacroGetCompilerName COMPILER_NAME_VAR)
  IF("${COMPILER_NAME_VAR}" STREQUAL "")
    MESSAGE(FATAL_ERROR "error: COMPILER_NAME_VAR CMake variable is empty !")
  ENDIF()
  
  SET(DART_COMPILER "${CMAKE_CXX_COMPILER}")
  IF(NOT DART_COMPILER)
    SET(DART_COMPILER "${CMAKE_C_COMPILER}")
  ENDIF()
  IF(NOT DART_COMPILER)
    SET(DART_COMPILER "unknown")
  ENDIF()
  IF(WIN32)
    SET(DART_NAME_COMPONENT "NAME_WE")
  ELSE()
    SET(DART_NAME_COMPONENT "NAME")
  ENDIF()
  IF(NOT BUILD_NAME_SYSTEM_NAME)
    SET(BUILD_NAME_SYSTEM_NAME "${CMAKE_SYSTEM_NAME}")
  ENDIF()
  IF(WIN32)
    SET(BUILD_NAME_SYSTEM_NAME "Win32")
  ENDIF()
  IF(UNIX OR BORLAND)
    GET_FILENAME_COMPONENT(DART_CXX_NAME 
      "${CMAKE_CXX_COMPILER}" ${DART_NAME_COMPONENT})
  ELSE()
    GET_FILENAME_COMPONENT(DART_CXX_NAME 
      "${CMAKE_BUILD_TOOL}" ${DART_NAME_COMPONENT})
  ENDIF()
  IF(DART_CXX_NAME MATCHES "msdev")
    SET(DART_CXX_NAME "vs60")
  ENDIF()
  IF(DART_CXX_NAME MATCHES "devenv")
    GET_VS_VERSION_STRING("${CMAKE_GENERATOR}" DART_CXX_NAME)
  ENDIF()
  STRING(REPLACE "c++" "g++" DART_CXX_NAME ${DART_CXX_NAME})
  SET(${COMPILER_NAME_VAR} ${DART_CXX_NAME})
ENDMACRO()

