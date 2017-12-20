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
#  This file was originally developed by
#   Dave Partyka and Jean-Christophe Fillion-Robin, Kitware Inc.
#  and was partially funded by NIH grant 3P41RR013218-12S1
#
################################################################################

#-----------------------------------------------------------------------------
# CMake https support
#-----------------------------------------------------------------------------
include(SlicerCheckCMakeHTTPS)

#-----------------------------------------------------------------------------
# Git protocol option
#-----------------------------------------------------------------------------
option(${CMAKE_PROJECT_NAME}_USE_GIT_PROTOCOL "If behind a firewall turn this off to use http instead." ON)
set(git_protocol "git")
if(NOT ${CMAKE_PROJECT_NAME}_USE_GIT_PROTOCOL)
  set(git_protocol "http")

  # Verify that the global git config has been updated with the expected "insteadOf" option.
  function(_check_for_required_git_config_insteadof base insteadof)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} config --global --get "url.${base}.insteadof"
      OUTPUT_VARIABLE output
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE error_code
      )
    if(error_code OR NOT "${output}" STREQUAL "${insteadof}")
      message(FATAL_ERROR
"Since the ExternalProject modules doesn't provide a mechanism to customize the clone step by "
"adding 'git config' statement between the 'git checkout' and the 'submodule init', it is required "
"to manually update your global git config to successfully build ${CMAKE_PROJECT_NAME} with "
"option ${CMAKE_PROJECT_NAME}_USE_GIT_PROTOCOL set to FALSE. "
"See http://na-mic.org/Mantis/view.php?id=2731"
"\nYou could do so by running the command:\n"
"  ${GIT_EXECUTABLE} config --global url.\"${base}\".insteadOf \"${insteadof}\"\n")
    endif()
  endfunction()

  if("${ITK_VERSION_MAJOR}" LESS 4)
    _check_for_required_git_config_insteadof("http://itk.org/" "git://itk.org/")
  endif()

endif()

#-----------------------------------------------------------------------------
# Enable and setup External project global properties
#-----------------------------------------------------------------------------

set(ep_common_c_flags "${CMAKE_C_FLAGS_INIT} ${ADDITIONAL_C_FLAGS}")
set(ep_common_cxx_flags "${CMAKE_CXX_FLAGS_INIT} ${ADDITIONAL_CXX_FLAGS}")

#-----------------------------------------------------------------------------
# Define list of additional options used to configure Slicer
#------------------------------------------------------------------------------

if(DEFINED CTEST_CONFIGURATION_TYPE)
  mark_as_superbuild(CTEST_CONFIGURATION_TYPE)
endif()

if(DEFINED CMAKE_CONFIGURATION_TYPES)
  mark_as_superbuild(CMAKE_CONFIGURATION_TYPES)
endif()

if(DEFINED SLICERAPP_LAUNCHER_SPLASHSCREEN_FILE)
  mark_as_superbuild(SLICERAPP_LAUNCHER_SPLASHSCREEN_FILE)
endif()

if(DEFINED SLICERAPP_WIN_ICON_FILE)
  mark_as_superbuild(SLICERAPP_WIN_ICON_FILE)
endif()

if(DEFINED SLICERAPP_APPLE_ICON_FILE)
  mark_as_superbuild(SLICERAPP_APPLE_ICON_FILE)
endif()

if(DEFINED SLICERAPP_LICENSE_FILE)
  mark_as_superbuild(SLICERAPP_LICENSE_FILE)
endif()

# Provide a mechanism to disable/enable one or more modules.
mark_as_superbuild(
  Slicer_QTLOADABLEMODULES_DISABLED:STRING
  Slicer_QTLOADABLEMODULES_ENABLED:STRING
  Slicer_QTSCRIPTEDMODULES_DISABLED:STRING
  Slicer_QTSCRIPTEDMODULES_ENABLED:STRING
  Slicer_CLIMODULES_DISABLED:STRING
  Slicer_CLIMODULES_ENABLED:STRING
  )

#------------------------------------------------------------------------------
# Slicer dependency list
#------------------------------------------------------------------------------

set(ITK_EXTERNAL_NAME ITKv4)

set(VTK_EXTERNAL_NAME VTKv9)

set(Slicer_DEPENDENCIES
  curl
  CTKAppLauncherLib
  teem
  ${VTK_EXTERNAL_NAME}
  ${ITK_EXTERNAL_NAME}
  CTK
  LibArchive
  RapidJSON
  )

set(CURL_ENABLE_SSL ${Slicer_USE_PYTHONQT_WITH_OPENSSL})

if(Slicer_USE_OpenIGTLink)
  list(APPEND Slicer_DEPENDENCIES OpenIGTLink)
endif()

if(Slicer_USE_SimpleITK)
  list(APPEND Slicer_DEPENDENCIES SimpleITK)
endif()

if(Slicer_BUILD_CLI_SUPPORT)
  list(APPEND Slicer_DEPENDENCIES SlicerExecutionModel)
endif()

if(Slicer_BUILD_EXTENSIONMANAGER_SUPPORT)
  list(APPEND Slicer_DEPENDENCIES qRestAPI)
endif()

if(Slicer_BUILD_DICOM_SUPPORT)
  list(APPEND Slicer_DEPENDENCIES DCMTK)
endif()

if(Slicer_BUILD_DICOM_SUPPORT AND Slicer_USE_PYTHONQT_WITH_OPENSSL)
  list(APPEND Slicer_DEPENDENCIES python-pydicom)
endif()

if(Slicer_USE_PYTHONQT AND Slicer_BUILD_EXTENSIONMANAGER_SUPPORT)
  list(APPEND Slicer_DEPENDENCIES
    python-chardet
    python-couchdb
    python-GitPython
    python-pip
    )
  if(Slicer_USE_PYTHONQT_WITH_OPENSSL OR Slicer_USE_SYSTEM_python)
    # python-PyGithub requires SSL support in Python
    list(APPEND Slicer_DEPENDENCIES python-PyGithub)
  else()
    message(STATUS "--------------------------------------------------")
    message(STATUS "Python was built without SSL support; "
                   "github integration will not be available. "
                   "Set Slicer_USE_PYTHONQT_WITH_OPENSSL=ON to enable this feature.")
    message(STATUS "--------------------------------------------------")
  endif()
endif()

if(Slicer_USE_CTKAPPLAUNCHER)
  list(APPEND Slicer_DEPENDENCIES CTKAPPLAUNCHER)
endif()

if(Slicer_USE_PYTHONQT)
  set(PYTHON_ENABLE_SSL ${Slicer_USE_PYTHONQT_WITH_OPENSSL})
  list(APPEND Slicer_DEPENDENCIES python)
endif()

if(Slicer_USE_NUMPY)
  list(APPEND Slicer_DEPENDENCIES NUMPY)
endif()

if(Slicer_USE_PYTHONQT_WITH_TCL AND UNIX)
  list(APPEND Slicer_DEPENDENCIES incrTcl)
endif()

#------------------------------------------------------------------------------
# Slicer_ADDITIONAL_DEPENDENCIES, EXTERNAL_PROJECT_ADDITIONAL_DIR
#------------------------------------------------------------------------------

#
# Setting the variable Slicer_ADDITIONAL_DEPENDENCIES allows to introduce additional
# Slicer external project dependencies.
#
# Additional external project files are looked up in the EXTERNAL_PROJECT_ADDITIONAL_DIR.
#

if(DEFINED Slicer_ADDITIONAL_DEPENDENCIES)
  list(APPEND Slicer_DEPENDENCIES ${Slicer_ADDITIONAL_DEPENDENCIES})
endif()

mark_as_superbuild(Slicer_DEPENDENCIES:STRING)

#------------------------------------------------------------------------------
# Include remote modules
#------------------------------------------------------------------------------
include(ExternalProjectAddSource)

macro(list_conditional_append cond list)
  if(${cond})
    list(APPEND ${list} ${ARGN})
  endif()
endmacro()

Slicer_Remote_Add(jqPlot
  URL http://slicer.kitware.com/midas3/download?items=15065&dummy=jquery.jqplot.1.0.4r1120.tar.gz
  URL_MD5 5c5d73730145c3963f09e1d3ca355580
  SOURCE_DIR_VAR jqPlot_DIR
  LABELS FIND_PACKAGE
  )
list(APPEND Slicer_REMOTE_DEPENDENCIES jqPlot)

Slicer_Remote_Add(OpenIGTLinkIF
  GIT_REPOSITORY ${git_protocol}://github.com/Slicer/OpenIGTLinkIF.git
  GIT_TAG 0266aa27ad19a00f8d2b1c04736b9f3d3fb25aee
  OPTION_NAME Slicer_BUILD_OpenIGTLinkIF
  OPTION_DEPENDS "Slicer_BUILD_QTLOADABLEMODULES;Slicer_USE_OpenIGTLink"
  LABELS REMOTE_MODULE
  )
list_conditional_append(Slicer_BUILD_OpenIGTLinkIF Slicer_REMOTE_DEPENDENCIES OpenIGTLinkIF)

option(Slicer_BUILD_MULTIVOLUME_SUPPORT "Build MultiVolume support." ON)
mark_as_advanced(Slicer_BUILD_MULTIVOLUME_SUPPORT)

Slicer_Remote_Add(MultiVolumeExplorer
  GIT_REPOSITORY ${git_protocol}://github.com/fedorov/MultiVolumeExplorer.git
  GIT_TAG b1bb131c41b6fa47a5c298b5d36b95b2e739d2e3
  OPTION_NAME Slicer_BUILD_MultiVolumeExplorer
  OPTION_DEPENDS "Slicer_BUILD_QTLOADABLEMODULES;Slicer_BUILD_MULTIVOLUME_SUPPORT;Slicer_USE_PYTHONQT"
  LABELS REMOTE_MODULE
  )
list_conditional_append(Slicer_BUILD_MultiVolumeExplorer Slicer_REMOTE_DEPENDENCIES MultiVolumeExplorer)

Slicer_Remote_Add(MultiVolumeImporter
  GIT_REPOSITORY ${git_protocol}://github.com/fedorov/MultiVolumeImporter.git
  GIT_TAG a85d4566f0b292773b3f28e06305e2e4d2cf3ca2
  OPTION_NAME Slicer_BUILD_MultiVolumeImporter
  OPTION_DEPENDS "Slicer_BUILD_QTLOADABLEMODULES;Slicer_BUILD_MULTIVOLUME_SUPPORT;Slicer_USE_PYTHONQT"
  LABELS REMOTE_MODULE
  )
list_conditional_append(Slicer_BUILD_MultiVolumeImporter Slicer_REMOTE_DEPENDENCIES MultiVolumeImporter)

Slicer_Remote_Add(SimpleFilters
  GIT_REPOSITORY ${git_protocol}://github.com/SimpleITK/SlicerSimpleFilters.git
  GIT_TAG 0e0648faeea0b3cbb8c27a93be0d95253ce13b98
  OPTION_NAME Slicer_BUILD_SimpleFilters
  OPTION_DEPENDS "Slicer_BUILD_QTSCRIPTEDMODULES;Slicer_USE_SimpleITK"
  LABELS REMOTE_MODULE
  )
list_conditional_append(Slicer_BUILD_SimpleFilters Slicer_REMOTE_DEPENDENCIES SimpleFilters)

set(BRAINSTools_options
  BRAINSTools_SUPERBUILD:BOOL=OFF
  BRAINSTools_CLI_LIBRARY_OUTPUT_DIRECTORY:PATH=${CMAKE_BINARY_DIR}/${Slicer_BINARY_INNER_SUBDIR}/${Slicer_CLIMODULES_LIB_DIR}
  BRAINSTools_CLI_ARCHIVE_OUTPUT_DIRECTORY:PATH=${CMAKE_BINARY_DIR}/${Slicer_BINARY_INNER_SUBDIR}/${Slicer_CLIMODULES_LIB_DIR}
  BRAINSTools_CLI_RUNTIME_OUTPUT_DIRECTORY:PATH=${CMAKE_BINARY_DIR}/${Slicer_BINARY_INNER_SUBDIR}/${Slicer_CLIMODULES_BIN_DIR}
  BRAINSTools_CLI_INSTALL_LIBRARY_DESTINATION:PATH=${Slicer_INSTALL_CLIMODULES_LIB_DIR}
  BRAINSTools_CLI_INSTALL_ARCHIVE_DESTINATION:PATH=${Slicer_INSTALL_CLIMODULES_LIB_DIR}
  BRAINSTools_CLI_INSTALL_RUNTIME_DESTINATION:PATH=${Slicer_INSTALL_CLIMODULES_BIN_DIR}
  BRAINSTools_DISABLE_TESTING:BOOL=ON
  USE_BRAINSFit:BOOL=ON
  USE_BRAINSROIAuto:BOOL=ON
  USE_BRAINSResample:BOOL=ON
  USE_BRAINSDemonWarp:BOOL=ON
  # BRAINSTools comes with some extra tool that should not be compiled by default
  USE_AutoWorkup:BOOL=OFF
  USE_ReferenceAtlas:BOOL=OFF
  USE_ANTS:BOOL=OFF
  USE_GTRACT:BOOL=OFF
  USE_BRAINSABC:BOOL=OFF
  USE_BRAINSTalairach:BOOL=OFF
  USE_BRAINSConstellationDetector:BOOL=OFF
  USE_BRAINSMush:BOOL=OFF
  USE_BRAINSInitializedControlPoints:BOOL=OFF
  USE_BRAINSMultiModeSegment:BOOL=OFF
  USE_BRAINSCut:BOOL=OFF
  USE_BRAINSLandmarkInitializer:BOOL=OFF
  USE_ImageCalculator:BOOL=OFF
  USE_BRAINSSurfaceTools:BOOL=OFF
  USE_BRAINSSnapShotWriter:BOOL=OFF
  USE_ConvertBetweenFileFormats:BOOL=OFF
  USE_BRAINSMultiSTAPLE:BOOL=OFF
  USE_BRAINSCreateLabelMapFromProbabilityMaps:BOOL=OFF
  USE_BRAINSContinuousClass:BOOL=OFF
  USE_ICCDEF:BOOL=OFF
  USE_BRAINSPosteriorToContinuousClass:BOOL=OFF
  USE_DebugImageViewer:BOOL=OFF
  BRAINS_DEBUG_IMAGE_WRITE:BOOL=OFF
  USE_BRAINSTransformConvert:BOOL=ON
  USE_DWIConvert:BOOL=${Slicer_BUILD_DICOM_SUPPORT} ## Need to figure out library linking
  USE_BRAINSDemonWarp:BOOL=ON
  USE_BRAINSRefacer:BOOL=OFF
  )
Slicer_Remote_Add(BRAINSTools
  GIT_REPOSITORY "${git_protocol}://github.com/Slicer/BRAINSTools.git"
  GIT_TAG "c1289e6686f3fd27db2ce0ea19c36e9792e40d2a" # master (from 2017-11-29, post v4.7.1)
  OPTION_NAME Slicer_BUILD_BRAINSTOOLS
  OPTION_DEPENDS "Slicer_BUILD_CLI_SUPPORT;Slicer_BUILD_CLI"
  LABELS REMOTE_MODULE
  VARS ${BRAINSTools_options}
  )
list_conditional_append(Slicer_BUILD_BRAINSTOOLS Slicer_REMOTE_DEPENDENCIES BRAINSTools)
if(Slicer_BUILD_BRAINSTOOLS)
  # This is added to SlicerConfig and is useful for extension depending on BRAINSTools
  set(BRAINSCommonLib_DIR "${Slicer_BINARY_DIR}/${Slicer_BINARY_INNER_SUBDIR}/Modules/Remote/BRAINSTools/BRAINSCommonLib")
  mark_as_superbuild(BRAINSCommonLib_DIR:PATH)
endif()

Slicer_Remote_Add(EMSegment
  SVN_REPOSITORY "http://svn.slicer.org/Slicer3/branches/Slicer4-EMSegment"
  SVN_REVISION -r "17142"
  OPTION_NAME Slicer_BUILD_EMSegment
  OPTION_DEPENDS "Slicer_BUILD_BRAINSTOOLS;Slicer_BUILD_QTLOADABLEMODULES;Slicer_USE_PYTHONQT_WITH_TCL"
  LABELS REMOTE_MODULE
  )
list_conditional_append(Slicer_BUILD_EMSegment Slicer_REMOTE_DEPENDENCIES EMSegment)

Slicer_Remote_Add(OtsuThresholdImageFilter
  GIT_REPOSITORY "${git_protocol}://github.com/Slicer/Slicer-OtsuThresholdImageFilter"
  GIT_TAG "cf39e5064472af31809ec1fa2f93fb97dc9a606e"
  OPTION_NAME Slicer_BUILD_OtsuThresholdImageFilter
  OPTION_DEPENDS "Slicer_BUILD_EMSegment"
  LABELS REMOTE_MODULE
  )
list_conditional_append(Slicer_BUILD_OtsuThresholdImageFilter Slicer_REMOTE_DEPENDENCIES OtsuThresholdImageFilter)

Slicer_Remote_Add(DataStore
  GIT_REPOSITORY "${git_protocol}://github.com/Slicer/Slicer-DataStore"
  GIT_TAG "57c6cb725344b5ec7a44a9644109b4589237d119"
  OPTION_NAME Slicer_BUILD_DataStore
  LABELS REMOTE_MODULE
  )
list_conditional_append(Slicer_BUILD_DataStore Slicer_REMOTE_DEPENDENCIES DataStore)

Slicer_Remote_Add(CompareVolumes
  GIT_REPOSITORY "${git_protocol}://github.com/pieper/CompareVolumes"
  GIT_TAG "c402286bf8dcb4a050a055bfd96548eb9876b485"
  OPTION_NAME Slicer_BUILD_CompareVolumes
  OPTION_DEPENDS "Slicer_USE_PYTHONQT"
  LABELS REMOTE_MODULE
  )
list_conditional_append(Slicer_BUILD_CompareVolumes Slicer_REMOTE_DEPENDENCIES CompareVolumes)

Slicer_Remote_Add(LandmarkRegistration
  GIT_REPOSITORY "${git_protocol}://github.com/pieper/LandmarkRegistration"
  GIT_TAG "a8c80bccea13979a98e967ed226a13d123a9c7ac"
  OPTION_NAME Slicer_BUILD_LandmarkRegistration
  OPTION_DEPENDS "Slicer_BUILD_CompareVolumes;Slicer_USE_PYTHONQT"
  LABELS REMOTE_MODULE
  )
list_conditional_append(Slicer_BUILD_LandmarkRegistration Slicer_REMOTE_DEPENDENCIES LandmarkRegistration)


#------------------------------------------------------------------------------
# Slicer_ADDITIONAL_PROJECTS
#------------------------------------------------------------------------------

#
# List of <proj>_DIR variables that will be passed to the inner build.
# Then, the variables are:
# (1) associated with CPACK_INSTALL_CMAKE_PROJECTS in SlicerCPack
# (2) used to get <proj>_LIBRARY_DIRS and update "libs_path" in SlicerCPackBundleFixup.
#

list(APPEND Slicer_ADDITIONAL_PROJECTS ${Slicer_ADDITIONAL_DEPENDENCIES})
if(Slicer_ADDITIONAL_PROJECTS)
  list(REMOVE_DUPLICATES Slicer_ADDITIONAL_PROJECTS)
  foreach(additional_project ${Slicer_ADDITIONAL_PROJECTS})
    # needed to do find_package within Slicer
    mark_as_superbuild(${additional_project}_DIR:PATH)
  endforeach()
  mark_as_superbuild(Slicer_ADDITIONAL_PROJECTS:STRING)
endif()

#------------------------------------------------------------------------------
# Process external projects, aggregate variable marked as superbuild and set <proj>_EP_ARGS variable.
#------------------------------------------------------------------------------

ExternalProject_Include_Dependencies(Slicer DEPENDS_VAR Slicer_DEPENDENCIES)

#------------------------------------------------------------------------------
# Define list of additional options used to configure Slicer
#------------------------------------------------------------------------------

set(EXTERNAL_PROJECT_OPTIONAL_ARGS)
if(WIN32)
  list(APPEND EXTERNAL_PROJECT_OPTIONAL_ARGS -DSlicer_SKIP_ROOT_DIR_MAX_LENGTH_CHECK:BOOL=ON)
endif()

#------------------------------------------------------------------------------
# Customizing SlicerApp metadata
#------------------------------------------------------------------------------

# Configuring Slicer setting these variables allows to overwrite the properties
# associated with the SlicerApp application.

foreach(name IN ITEMS
  DESCRIPTION_SUMMARY
  DESCRIPTION_FILE
  LAUNCHER_SPLASHSCREEN_FILE
  APPLE_ICON_FILE
  WIN_ICON_FILE
  LICENSE_FILE
  )
  if(DEFINED SlicerApp_${name})
    list(APPEND EXTERNAL_PROJECT_OPTIONAL_ARGS
      -DSlicerApp_${name}:STRING=${SlicerApp_${name}}
      )
  endif()
endforeach()

#------------------------------------------------------------------------------
# Slicer_EXTENSION_SOURCE_DIRS
#------------------------------------------------------------------------------

#
# Configuring Slicer using
#
#   cmake -DSlicer_EXTENSION_SOURCE_DIRS:STRING=/path/to/ExtensionA;/path/to/ExtensionB [...] /path/to/source/Slicer
#
# will ensure the source of each extensions are *built* by Slicer. This is done
# as part of the Slicer inner build by adding each directory in the top-level CMakeLists.txt.
#
# Note that using 'Slicer_Remote_Add' specifying the label 'REMOTE_EXTENSION' (see above)
# will checkout the extension sources and append the corresponding source directory to
# the variable Slicer_EXTENSION_SOURCE_DIRS.
#

#------------------------------------------------------------------------------
# Slicer_EXTENSION_INSTALL_DIRS
#------------------------------------------------------------------------------

#
# Configuring Slicer using
#
#   cmake -DSlicer_EXTENSION_INSTALL_DIRS:STRING=/path/to/ExtensionA-install-tree;/path/to/ExtensionB-install-tree [...] /path/to/source/Slicer
#
# will ensure the content of each extensions install directories are *packaged*
# with Slicer.
#
# Corresponding install rules are found in "CMake/SlicerBlockInstallExtensionPackages.cmake"
#

#------------------------------------------------------------------------------
# Configure and build Slicer
#------------------------------------------------------------------------------
set(proj Slicer)

ExternalProject_Add(${proj}
  ${${proj}_EP_ARGS}
  DEPENDS ${Slicer_DEPENDENCIES} ${Slicer_REMOTE_DEPENDENCIES}
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
  BINARY_DIR ${CMAKE_BINARY_DIR}/${Slicer_BINARY_INNER_SUBDIR}
  DOWNLOAD_COMMAND ""
  UPDATE_COMMAND ""
  CMAKE_CACHE_ARGS
    -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
    -DCMAKE_CXX_FLAGS:STRING=${ep_common_cxx_flags}
    -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
    -DCMAKE_C_FLAGS:STRING=${ep_common_c_flags}
    -DCMAKE_CXX_STANDARD:STRING=${CMAKE_CXX_STANDARD}
    -DCMAKE_CXX_STANDARD_REQUIRED:BOOL=${CMAKE_CXX_STANDARD_REQUIRED}
    -DCMAKE_CXX_EXTENSIONS:BOOL=${CMAKE_CXX_EXTENSIONS}
    -DADDITIONAL_C_FLAGS:STRING=${ADDITIONAL_C_FLAGS}
    -DADDITIONAL_CXX_FLAGS:STRING=${ADDITIONAL_CXX_FLAGS}
    -DSlicer_REQUIRED_C_FLAGS:STRING=${Slicer_REQUIRED_C_FLAGS}
    -DSlicer_REQUIRED_CXX_FLAGS:STRING=${Slicer_REQUIRED_CXX_FLAGS}
    -DSlicer_SUPERBUILD:BOOL=OFF
    -DSlicer_SUPERBUILD_DIR:PATH=${CMAKE_BINARY_DIR}
    -D${Slicer_MAIN_PROJECT}_APPLICATION_NAME:STRING=${${Slicer_MAIN_PROJECT}_APPLICATION_NAME}
    -DSlicer_EXTENSION_SOURCE_DIRS:STRING=${Slicer_EXTENSION_SOURCE_DIRS}
    -DSlicer_EXTENSION_INSTALL_DIRS:STRING=${Slicer_EXTENSION_INSTALL_DIRS}
    -DExternalData_OBJECT_STORES:PATH=${ExternalData_OBJECT_STORES}
    ${EXTERNAL_PROJECT_OPTIONAL_ARGS}
  INSTALL_COMMAND ""
  )

ExternalProject_AlwaysConfigure(${proj})
