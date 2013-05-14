
# Make sure this file is included only once
get_filename_component(CMAKE_CURRENT_LIST_FILENAME ${CMAKE_CURRENT_LIST_FILE} NAME_WE)
if(${CMAKE_CURRENT_LIST_FILENAME}_FILE_INCLUDED)
  return()
endif()
set(${CMAKE_CURRENT_LIST_FILENAME}_FILE_INCLUDED 1)

# Set dependency list
set(OpenIGTLink_DEPENDENCIES "")

# Include dependent projects if any
SlicerMacroCheckExternalProjectDependency(OpenIGTLink)
set(proj OpenIGTLink)

set(EXTERNAL_PROJECT_OPTIONAL_ARGS)

# Set CMake OSX variable to pass down the external project
if(APPLE)
  list(APPEND EXTERNAL_PROJECT_OPTIONAL_ARGS
    -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
    -DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT}
    -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET})
endif()

if(NOT CMAKE_CONFIGURATION_TYPES)
  list(APPEND EXTERNAL_PROJECT_OPTIONAL_ARGS
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE})
endif()

set(CMAKE_PROJECT_INCLUDE_EXTERNAL_PROJECT_ARG)
if(CTEST_USE_LAUNCHERS)
  set(CMAKE_PROJECT_INCLUDE_EXTERNAL_PROJECT_ARG
    "-DCMAKE_PROJECT_OpenIGTLink_INCLUDE:FILEPATH=${CMAKE_ROOT}/Modules/CTestUseLaunchers.cmake")
endif()

#message(STATUS "${__indent}Adding project ${proj}")

ExternalProject_Add(${proj}
  GIT_REPOSITORY "${git_protocol}://github.com/openigtlink/OpenIGTLink.git"
  GIT_TAG "66e272daa0744cbcdd492fb02137b19acff33019"
  "${${PROJECT_NAME}_EP_UPDATE_IF_GREATER_288}"
  SOURCE_DIR OpenIGTLink
  BINARY_DIR OpenIGTLink-build
  CMAKE_GENERATOR ${gen}
  CMAKE_ARGS
    -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
    -DCMAKE_CXX_FLAGS:STRING=${ep_common_cxx_flags}
    -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
    -DCMAKE_C_FLAGS:STRING=${ep_common_c_flags}
    ${CMAKE_PROJECT_INCLUDE_EXTERNAL_PROJECT_ARG}
    -DBUILD_TESTING:BOOL=OFF
    -DBUILD_SHARED_LIBS:BOOL=ON
    -DOpenIGTLink_PROTOCOL_VERSION_2:BOOL=ON
    ${EXTERNAL_PROJECT_OPTIONAL_ARGS}
  INSTALL_COMMAND ""
  DEPENDS
    ${OpenIGTLink_DEPENDENCIES}
  )

set(OpenIGTLink_DIR ${CMAKE_BINARY_DIR}/OpenIGTLink-build)
