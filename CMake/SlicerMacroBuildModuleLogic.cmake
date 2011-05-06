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
# SlicerMacroBuildModuleLogic
#

MACRO(SlicerMacroBuildModuleLogic)
  SLICER_PARSE_ARGUMENTS(MODULELOGIC
    "NAME;EXPORT_DIRECTIVE;SRCS;INCLUDE_DIRECTORIES;TARGET_LIBRARIES"
    "DISABLE_WRAP_PYTHON"
    ${ARGN}
    )
  
  LIST(APPEND MODULELOGIC_INCLUDE_DIRECTORIES
    ${Slicer_Libs_INCLUDE_DIRS}
    ${Slicer_Base_INCLUDE_DIRS}
    ${Slicer_ModuleLogic_INCLUDE_DIRS}
    ${Slicer_ModuleMRML_INCLUDE_DIRS}
    )
  
  # Note: Linking against qSlicerBaseQTCLI provides logic with 
  #       access to the core application modulemanager. Using the module manager
  #       a module logic can then use the services provided by registrered 
  #       command line module (CLI).
    
  LIST(APPEND MODULELOGIC_TARGET_LIBRARIES
    qSlicerBaseQTCLI
    )
  
  SlicerMacroBuildModuleLibrary(
    NAME ${MODULELOGIC_NAME}
    EXPORT_DIRECTIVE ${MODULELOGIC_EXPORT_DIRECTIVE}
    SRCS ${MODULELOGIC_SRCS}
    INCLUDE_DIRECTORIES ${MODULELOGIC_INCLUDE_DIRECTORIES}
    TARGET_LIBRARIES ${MODULELOGIC_TARGET_LIBRARIES}
    )

  #-----------------------------------------------------------------------------
  # Update Slicer_ModuleLogic_INCLUDE_DIRS
  #-----------------------------------------------------------------------------
  IF(Slicer_SOURCE_DIR)
    SET(Slicer_ModuleLogic_INCLUDE_DIRS
      ${Slicer_ModuleLogic_INCLUDE_DIRS}
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_BINARY_DIR}
      CACHE INTERNAL "Slicer Module logic includes" FORCE)
  ENDIF()
  
  # --------------------------------------------------------------------------
  # Python wrapping
  # --------------------------------------------------------------------------
  IF(NOT ${MODULELOGIC_DISABLE_WRAP_PYTHON} AND VTK_WRAP_PYTHON)

    SET(Slicer_Wrapped_LIBRARIES
      SlicerBaseLogicPythonD
      )
    
    SlicerMacroPythonWrapModuleLibrary(
      NAME ${MODULELOGIC_NAME}
      SRCS ${MODULELOGIC_SRCS}
      WRAPPED_TARGET_LIBRARIES ${Slicer_Wrapped_LIBRARIES}
      RELATIVE_PYTHON_DIR "slicer/modulelogic"
      )

  ENDIF()
    
ENDMACRO()
