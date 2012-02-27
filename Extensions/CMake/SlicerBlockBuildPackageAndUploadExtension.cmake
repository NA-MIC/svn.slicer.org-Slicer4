
#-----------------------------------------------------------------------------
# Extract dashboard option passed from command line
#-----------------------------------------------------------------------------
# Note: The syntax to pass option from the command line while invoking ctest is
#       the following: ctest -S /path/to/script.cmake,OPTNAME1##OPTVALUE1^^OPTNAME2##OPTVALUE2
#
# Example:
#       ctest -S /path/to/script.cmake,SCRIPT_MODE##continuous^^GIT_TAG##next
#
if(NOT CTEST_SCRIPT_ARG STREQUAL "")
  cmake_policy(PUSH)
  cmake_policy(SET CMP0007 OLD)
  string(REPLACE "^^" "\\;" CTEST_SCRIPT_ARG_AS_LIST "${CTEST_SCRIPT_ARG}")
  set(CTEST_SCRIPT_ARG_AS_LIST ${CTEST_SCRIPT_ARG_AS_LIST})
  foreach(argn_argv ${CTEST_SCRIPT_ARG_AS_LIST})
    string(REPLACE "##" "\\;" argn_argv_list ${argn_argv})
    set(argn_argv_list ${argn_argv_list})
    list(LENGTH argn_argv_list argn_argv_list_length)
    list(GET argn_argv_list 0 argn)
    if(argn_argv_list_length GREATER 1)
      list(REMOVE_AT argn_argv_list 0) # Take first item
      set(argv) # Convert from list to string separated by '='
      foreach(str_item ${argn_argv_list})
        set(argv "${argv}=${str_item}")
      endforeach()
      string(SUBSTRING ${argv} 1 -1 argv) # Remove first unwanted '='
      string(REPLACE "/-/" "//" argv ${argv}) # See http://www.cmake.org/Bug/view.php?id=12953
      string(REPLACE "-AMP-" "&" argv ${argv})
      string(REPLACE "-WHT-" "?" argv ${argv})
      set(${argn} ${argv})
    endif()
  endforeach()
  cmake_policy(POP)
endif()

#-----------------------------------------------------------------------------
# Macro allowing to set a variable to its default value only if not already defined
macro(setIfNotDefined var defaultvalue)
  if(NOT DEFINED ${var})
    set(${var} "${defaultvalue}")
  endif()
endmacro()

#-----------------------------------------------------------------------------
# Set build configuration
if(NOT "${CMAKE_CFG_INTDIR}" STREQUAL ".")
  set(CTEST_BUILD_CONFIGURATION ${CMAKE_CFG_INTDIR})
else()
  set(CTEST_BUILD_CONFIGURATION ${CMAKE_BUILD_TYPE})
endif()

#-----------------------------------------------------------------------------
# Sanity checks
set(expected_defined_vars EXTENSION_NAME EXTENSION_CATEGORY EXTENSION_DESCRIPTION EXTENSION_HOMEPAGE EXTENSION_SOURCE_DIR EXTENSION_SUPERBUILD_BINARY_DIR EXTENSION_BUILD_SUBDIRECTORY EXTENSION_ENABLED CTEST_BUILD_CONFIGURATION CTEST_CMAKE_GENERATOR Slicer_CMAKE_DIR Slicer_DIR Slicer_WC_REVISION EXTENSION_BUILD_OPTIONS_STRING RUN_CTEST_CONFIGURE RUN_CTEST_BUILD RUN_CTEST_TEST RUN_CTEST_PACKAGES RUN_CTEST_SUBMIT RUN_CTEST_UPLOAD BUILD_TESTING)
if(RUN_CTEST_UPLOAD)
  list(APPEND expected_defined_vars
    MIDAS_PACKAGE_URL MIDAS_PACKAGE_EMAIL MIDAS_PACKAGE_API_KEY
    EXTENSION_ARCHITECTURE EXTENSION_BITNESS EXTENSION_OPERATING_SYSTEM
    )
endif()

foreach(var ${expected_defined_vars})
  if(NOT DEFINED ${var})
    message(FATAL_ERROR "Variable ${var} is not defined !")
  endif()
endforeach()

set(CMAKE_MODULE_PATH
  ${Slicer_CMAKE_DIR}
  ${Slicer_CMAKE_DIR}/../Extensions/CMake
  ${CMAKE_MODULE_PATH}
  )

include(CMakeParseArguments)
include(${Slicer_CMAKE_DIR}/SlicerFunctionCTestPackage.cmake)
include(${Slicer_CMAKE_DIR}/SlicerFunctionMIDASCTestUploadURL.cmake)
include(${Slicer_CMAKE_DIR}/../Extensions/CMake/SlicerFunctionMIDASUploadExtension.cmake)

#-----------------------------------------------------------------------------
# Set site name
site_name(CTEST_SITE)
# Force to lower case
string(TOLOWER "${CTEST_SITE}" CTEST_SITE)

# Set build name
set(CTEST_BUILD_NAME "${Slicer_WC_REVISION}-${EXTENSION_NAME}-${EXTENSION_COMPILER}-${EXTENSION_BUILD_OPTIONS_STRING}-${CTEST_BUILD_CONFIGURATION}")

setIfNotDefined(CTEST_PARALLEL_LEVEL 8)
setIfNotDefined(CTEST_MODEL "Experimental")

set(label ${EXTENSION_NAME})
set_property(GLOBAL PROPERTY SubProject ${label})
set_property(GLOBAL PROPERTY Label ${label})

# If no CTestConfig.cmake file is found in ${ctestconfig_dest_dir},
# one will be generated.
set(ctestconfig_dest_dir ${EXTENSION_SUPERBUILD_BINARY_DIR}/${EXTENSION_BUILD_SUBDIRECTORY})
if(${CMAKE_VERSION} VERSION_LESS "2.8.7")
  set(ctestconfig_dest_dir ${EXTENSION_SOURCE_DIR})
endif()
if(NOT EXISTS ${ctestconfig_dest_dir}/CTestConfig.cmake)
  message(STATUS "CTestConfig.cmake has been written to: ${ctestconfig_dest_dir}")
  file(WRITE ${ctestconfig_dest_dir}/CTestConfig.cmake
"set(CTEST_PROJECT_NAME \"Slicer\")
set(CTEST_NIGHTLY_START_TIME \"3:00:00 UTC\")

set(CTEST_DROP_METHOD \"http\")
set(CTEST_DROP_SITE \"slicer.cdash.org\")
set(CTEST_DROP_LOCATION \"/submit.php?project=Slicer4\")
set(CTEST_DROP_SITE_CDASH TRUE)")
endif()

set(track "Extensions-${CTEST_MODEL}")
ctest_start(${CTEST_MODEL} TRACK ${track} ${EXTENSION_SOURCE_DIR} ${EXTENSION_SUPERBUILD_BINARY_DIR})
ctest_read_custom_files(${EXTENSION_SUPERBUILD_BINARY_DIR} ${EXTENSION_SUPERBUILD_BINARY_DIR}/${EXTENSION_BUILD_SUBDIRECTORY})

set(cmakecache_content
"#Generated by SlicerBlockBuildPackageAndUploadExtension.cmake
CMAKE_BUILD_TYPE:STRING=${CTEST_BUILD_CONFIGURATION}
Slicer_DIR:PATH=${Slicer_DIR}
MIDAS_PACKAGE_URL:STRING=${MIDAS_PACKAGE_URL}
MIDAS_PACKAGE_EMAIL:STRING=${MIDAS_PACKAGE_EMAIL}
MIDAS_PACKAGE_API_KEY:STRING=${MIDAS_PACKAGE_API_KEY}
")

#-----------------------------------------------------------------------------
# Write CMakeCache.txt only if required
set(cmakecache_current "")
if(EXISTS ${EXTENSION_SUPERBUILD_BINARY_DIR}/CMakeCache.txt)
  file(READ ${EXTENSION_SUPERBUILD_BINARY_DIR}/CMakeCache.txt cmakecache_current)
endif()
if(NOT ${cmakecache_content} STREQUAL "${cmakecache_current}")
  file(WRITE ${EXTENSION_SUPERBUILD_BINARY_DIR}/CMakeCache.txt ${cmakecache_content})
endif()

# Explicitly set CTEST_BINARY_DIRECTORY so that ctest_submit find
# the xml part files in <EXTENSION_SUPERBUILD_BINARY_DIR>/Testing
set(CTEST_BINARY_DIRECTORY ${EXTENSION_SUPERBUILD_BINARY_DIR})

#-----------------------------------------------------------------------------
# Configure extension
if(RUN_CTEST_CONFIGURE)
  #message("----------- [ Configuring extension ${EXTENSION_NAME} ] -----------")
  ctest_configure(
    BUILD ${EXTENSION_SUPERBUILD_BINARY_DIR}
    SOURCE ${EXTENSION_SOURCE_DIR}
    )
  if(RUN_CTEST_SUBMIT)
    ctest_submit(PARTS Configure)
  endif()
endif()

#-----------------------------------------------------------------------------
# Build extension
set(build_errors)
if(RUN_CTEST_BUILD)
  #message("----------- [ Building extension ${EXTENSION_NAME} ] -----------")
  ctest_build(BUILD ${EXTENSION_SUPERBUILD_BINARY_DIR} NUMBER_ERRORS build_errors APPEND)
  if(RUN_CTEST_SUBMIT)
    ctest_submit(PARTS Build)
  endif()
endif()

#-----------------------------------------------------------------------------
# Test extension
if(BUILD_TESTING AND RUN_CTEST_TEST)
  #message("----------- [ Testing extension ${EXTENSION_NAME} ] -----------")
  # Check if there are tests to run
  execute_process(COMMAND ${CMAKE_CTEST_COMMAND} -N
    WORKING_DIRECTORY ${EXTENSION_SUPERBUILD_BINARY_DIR}/${EXTENSION_BUILD_SUBDIRECTORY}
    OUTPUT_VARIABLE output
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  string(REGEX REPLACE ".*Total Tests: ([0-9]+)" "\\1" test_count "${output}")
  if("${test_count}" GREATER 0)
    ctest_test(
        BUILD ${EXTENSION_SUPERBUILD_BINARY_DIR}/${EXTENSION_BUILD_SUBDIRECTORY}
        PARALLEL_LEVEL ${CTEST_PARALLEL_LEVEL})
    if(RUN_CTEST_SUBMIT)
      ctest_submit(PARTS Test)
    endif()
  endif()
endif()

#-----------------------------------------------------------------------------
# Package extension
if(RUN_CTEST_PACKAGES)
  if(build_errors GREATER "0")
    message(WARNING "Skip extension packaging: ${build_errors} build error(s) occured !")
  else()
    #message("----------- [ Packaging extension ${EXTENSION_NAME} ] -----------")
    message("Packaging extension ${EXTENSION_NAME} ...")
    set(extension_packages)
    SlicerFunctionCTestPackage(
      BINARY_DIR ${EXTENSION_SUPERBUILD_BINARY_DIR}/${EXTENSION_BUILD_SUBDIRECTORY}
      CONFIG ${CTEST_BUILD_CONFIGURATION}
      RETURN_VAR extension_packages)

    if(RUN_CTEST_UPLOAD AND COMMAND ctest_upload)
      message("Uploading extension ${EXTENSION_NAME} ...")

      # Update CMake module path so that our custom FindGit.cmake module is used.
      set(CMAKE_MODULE_PATH ${Slicer_CMAKE_DIR} ${CMAKE_MODULE_PATH})
      include(SlicerMacroExtractRepositoryInfo)
      SlicerMacroExtractRepositoryInfo(VAR_PREFIX EXTENSION SOURCE_DIR ${EXTENSION_SOURCE_DIR})

      foreach(p ${extension_packages})
        SlicerFunctionMIDASUploadExtension(
          SERVER_URL ${MIDAS_PACKAGE_URL}
          SERVER_EMAIL ${MIDAS_PACKAGE_EMAIL}
          SERVER_APIKEY ${MIDAS_PACKAGE_API_KEY}
          TMP_DIR ${EXTENSION_SUPERBUILD_BINARY_DIR}/${EXTENSION_BUILD_SUBDIRECTORY}
          SUBMISSION_TYPE ${CTEST_MODEL}
          SLICER_REVISION ${Slicer_WC_REVISION}
          EXTENSION_NAME ${EXTENSION_NAME}
          EXTENSION_CATEGORY ${EXTENSION_CATEGORY}
          EXTENSION_DESCRIPTION ${EXTENSION_DESCRIPTION}
          EXTENSION_HOMEPAGE ${EXTENSION_HOMEPAGE}
          EXTENSION_REPOSITORY_TYPE ${EXTENSION_WC_TYPE}
          EXTENSION_REPOSITORY_URL ${EXTENSION_WC_URL}
          EXTENSION_SOURCE_REVISION ${EXTENSION_WC_REVISION}
          EXTENSION_ENABLED ${EXTENSION_ENABLED}
          OPERATING_SYSTEM ${EXTENSION_OPERATING_SYSTEM}
          ARCHITECTURE ${EXTENSION_ARCHITECTURE}
          PACKAGE_FILEPATH ${p}
          PACKAGE_TYPE "archive"
          RELEASE ${release}
          RESULT_VARNAME slicer_midas_upload_status
          )
        if(NOT slicer_midas_upload_status STREQUAL "ok")
          ctest_upload(FILES ${p}) #on failure, upload the package to CDash instead
        else()
          SlicerFunctionMIDASCTestUploadURL(${p}) # on success, upload a link to CDash
        endif()
        if(RUN_CTEST_SUBMIT)
          ctest_submit(PARTS Upload)
        endif()
      endforeach()
    endif()
  endif()
endif()

