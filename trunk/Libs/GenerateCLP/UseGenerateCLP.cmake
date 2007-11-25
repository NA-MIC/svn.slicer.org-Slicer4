FIND_PACKAGE(ITK)
IF(ITK_FOUND)
  INCLUDE(${ITK_USE_FILE})
ELSE(ITK_FOUND)
  MESSAGE(ERROR "Cannot build without ITK. Please set ITK_DIR.")
ENDIF(ITK_FOUND)

#
# If being build as part of Slicer3, we know where to find tclap include files

IF(Slicer3_SOURCE_DIR)
  SET(TCLAP_DIR ${Slicer3_BINARY_DIR}/Libs/tclap)
ENDIF(Slicer3_SOURCE_DIR)

FIND_PACKAGE(TCLAP REQUIRED)
IF(TCLAP_FOUND)
   INCLUDE(${TCLAP_USE_FILE})
ENDIF(TCLAP_FOUND)

FIND_PACKAGE(ModuleDescriptionParser REQUIRED)
IF(ModuleDescriptionParser_FOUND)
   INCLUDE(${ModuleDescriptionParser_USE_FILE})
ENDIF(ModuleDescriptionParser_FOUND)

#INCLUDE_DIRECTORIES (${TCLAP_SOURCE_DIR}/include)
#
#IF(ModuleDescriptionParser_SOURCE_DIR)
#  INCLUDE_DIRECTORIES(
#  ${ModuleDescriptionParser_SOURCE_DIR}
#  ${ModuleDescriptionParser_BINARY_DIR}
#  )
#ELSE(ModuleDescriptionParser_SOURCE_DIR)
#  INCLUDE_DIRECTORIES(
#  ${Slicer3_SOURCE_DIR}/Libs/ModuleDescriptionParser
#  )
#ENDIF(ModuleDescriptionParser_SOURCE_DIR)

UTILITY_SOURCE(GENERATECLP_EXE GenerateCLP ./ GenerateCLP.cxx)
IF (NOT GENERATECLP_EXE)
  FIND_PROGRAM(GENERATECLP_EXE GenerateCLP PATHS ${GenerateCLP_BINARY_DIR} DOC "GenerateCLP executable")
  MESSAGE(ERROR " Requires GenerateCLP executable. Please specify its location.")
ENDIF (NOT GENERATECLP_EXE)

# create the .clp files
# usage: GENERATE_CLP(foo_SRCS XML_FILE [LOGO_FILE])
MACRO(GENERATECLP SOURCES)
    # what is the filename without the extension
    GET_FILENAME_COMPONENT(TMP_FILENAME ${ARGV1} NAME_WE)
        
    # the input file might be full path so handle that
    GET_FILENAME_COMPONENT(TMP_FILEPATH ${ARGV1} PATH)

    # compute the input filename
    IF (TMP_FILEPATH)
      SET(TMP_INPUT ${TMP_FILEPATH}/${TMP_FILENAME}.xml) 
    ELSE (TMP_FILEPATH)
      SET(TMP_INPUT ${CMAKE_CURRENT_SOURCE_DIR}/${TMP_FILENAME}.xml)
    ENDIF (TMP_FILEPATH)

    # add custom command to output
    IF ("x${ARGV2}" STREQUAL "x")
      ADD_CUSTOM_COMMAND(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TMP_FILENAME}CLP.h
        DEPENDS ${GENERATECLP_EXE} ${TMP_INPUT}
        COMMAND ${GENERATECLP_EXE}
          ${TMP_INPUT} ${CMAKE_CURRENT_BINARY_DIR}/${TMP_FILENAME}CLP.h
      )
      # mark the .clp file as a header file
      SET_SOURCE_FILES_PROPERTIES(${CMAKE_CURRENT_BINARY_DIR}/${TMP_FILENAME}CLP.h PROPERTIES HEADER_FILE_ONLY TRUE)
      SET_SOURCE_FILES_PROPERTIES(${TMP_FILENAME}.cxx PROPERTIES OBJECT_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TMP_FILENAME}CLP.h)
    ELSE ("x${ARGV2}" STREQUAL "x")
      ADD_CUSTOM_COMMAND(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TMP_FILENAME}CLP.h
        DEPENDS ${GENERATECLP_EXE} ${TMP_INPUT} ${ARGV2}
        COMMAND ${GENERATECLP_EXE} --logoFiles ${ARGV2}
          ${TMP_INPUT} ${CMAKE_CURRENT_BINARY_DIR}/${TMP_FILENAME}CLP.h
      )
      # mark the .clp file as a header file
      SET_SOURCE_FILES_PROPERTIES(${CMAKE_CURRENT_BINARY_DIR}/${TMP_FILENAME}CLP.h PROPERTIES HEADER_FILE_ONLY TRUE)
      # mark the logo include file as a header file
      SET_SOURCE_FILES_PROPERTIES(${ARGV2} PROPERTIES HEADER_FILE_ONLY TRUE)
      SET_SOURCE_FILES_PROPERTIES(${TMP_FILENAME}.cxx PROPERTIES OBJECT_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TMP_FILENAME}CLP.h OBJECT_DEPENDS ${ARGV2})
    ENDIF ("x${ARGV2}" STREQUAL "x")

    SET(${SOURCES} ${CMAKE_CURRENT_BINARY_DIR}/${TMP_FILENAME}CLP.h ${${SOURCES}}) 
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
ENDMACRO(GENERATECLP)
