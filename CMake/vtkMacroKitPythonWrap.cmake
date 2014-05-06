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

# Based on VTK/CMake/KitCommonWrapBlock.cmake

macro(vtkMacroKitPythonWrap)
  set(options)
  set(oneValueArgs KIT_NAME KIT_INSTALL_BIN_DIR KIT_INSTALL_LIB_DIR)
  set(multiValueArgs KIT_SRCS KIT_PYTHON_EXTRA_SRCS KIT_WRAP_HEADERS KIT_PYTHON_LIBRARIES)
  cmake_parse_arguments(MY "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # Sanity checks
  set(expected_defined_vars
    VTK_CMAKE_DIR VTK_WRAP_PYTHON BUILD_SHARED_LIBS VTK_LIBRARIES)
  foreach(var ${expected_defined_vars})
    if(NOT DEFINED ${var})
      message(FATAL_ERROR "error: ${var} CMake variable is not defined !")
    endif()
  endforeach()

  set(expected_nonempty_vars KIT_NAME KIT_INSTALL_BIN_DIR KIT_INSTALL_LIB_DIR)
  foreach(var ${expected_nonempty_vars})
    if("${MY_${var}}" STREQUAL "")
      message(FATAL_ERROR "error: ${var} CMake variable is empty !")
    endif()
  endforeach()

  if(VTK_WRAP_PYTHON AND BUILD_SHARED_LIBS)

    # Tell vtkWrapPython.cmake to set VTK_PYTHON_LIBRARIES for us.
    set(VTK_WRAP_PYTHON_FIND_LIBS 1)
    include(${VTK_CMAKE_DIR}/vtkWrapPython.cmake)

    VTK_WRAP_PYTHON3(${MY_KIT_NAME}Python KitPython_SRCS "${TMP_WRAP_FILES}")

    include_directories("${PYTHON_INCLUDE_PATH}")

    # Create a python module that can be loaded dynamically.  It links to
    # the shared library containing the wrappers for this kit.
    add_library(${MY_KIT_NAME}PythonD ${KitPython_SRCS} ${MY_KIT_PYTHON_EXTRA_SRCS})

    # Not all the vtk libraries have their python wrapping
    if(${VTK_VERSION_MAJOR} GREATER 5)
       set(VTK_NO_PYTHON_WRAP_LIBRARIES
         vtkexpat
         vtkexoIIc
         vtkjsoncpp
         vtkpng
         vtksys
         vtkNetCDF
         vtkNetCDF_cxx
         vtkhdf5_hl
         vtkhdf5
         vtkalglib
         vtkDICOMParser
         vtkmetaio
         vtkjpeg
         vtktiff
         vtkfreetype
         vtkftgl
         vtkgl2ps
         vtksqlite
         vtkoggtheora
         vtkWrappingPythonCore
         vtkWrappingTools
         vtkGUISupportQt
         vtklibxml2
         vtkproj4
         vtkViewsQt
         vtkGUISupportQtWebkit
         vtkGUISupportQtSQL
         vtkGUISupportQtOpenGL
         )
      if (NOT WIN32)
        list(APPEND VTK_NO_PYTHON_WRAP_LIBRARIES
          vtkRenderingFreeTypeFontConfig)
      endif()
    else()
      set(VTK_NO_PYTHON_WRAP_LIBRARIES "")
    endif()
    foreach(lib ${VTK_NO_PYTHON_WRAP_LIBRARIES})
      list(REMOVE_ITEM VTK_LIBRARIES ${lib})
    endforeach()

    set(VTK_KIT_PYTHON_LIBRARIES)
    foreach(c ${VTK_LIBRARIES})
      if(${c} MATCHES "^vtk.+") # exclude system libraries
        list(APPEND VTK_KIT_PYTHON_LIBRARIES ${c}PythonD)
      endif()
    endforeach()
    if(${VTK_VERSION_MAJOR} GREATER 5)
      set(VTK_PYTHON_CORE vtkWrappingPythonCore)
    else()
      set(VTK_PYTHON_CORE vtkPythonCore)
    endif()
    target_link_libraries(
      ${MY_KIT_NAME}PythonD
      ${MY_KIT_NAME}
      ${VTK_PYTHON_CORE}
      ${VTK_PYTHON_LIBRARIES}
      ${VTK_KIT_PYTHON_LIBRARIES}
      ${MY_KIT_PYTHON_LIBRARIES}
      )

    install(TARGETS ${MY_KIT_NAME}PythonD
      RUNTIME DESTINATION ${MY_KIT_INSTALL_BIN_DIR} COMPONENT RuntimeLibraries
      LIBRARY DESTINATION ${MY_KIT_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${MY_KIT_INSTALL_LIB_DIR} COMPONENT Development
      )

    # Add a top-level dependency on the main kit library.  This is needed
    # to make sure no python source files are generated until the
    # hierarchy file is built (it is built when the kit library builds)
    add_dependencies(${MY_KIT_NAME}PythonD ${MY_KIT_NAME})

    # Add dependencies that may have been generated by VTK_WRAP_PYTHON3 to
    # the python wrapper library.  This is needed for the
    # pre-custom-command hack in Visual Studio 6.
    if(KIT_PYTHON_DEPS)
      add_dependencies(${MY_KIT_NAME}PythonD ${KIT_PYTHON_DEPS})
    endif()

    # Create a python module that can be loaded dynamically.  It links to
    # the shared library containing the wrappers for this kit.
    add_library(${MY_KIT_NAME}Python MODULE ${MY_KIT_NAME}PythonInit.cxx)
    target_link_libraries(${MY_KIT_NAME}Python ${MY_KIT_NAME}PythonD)

    # Python extension modules on Windows must have the extension ".pyd"
    # instead of ".dll" as of Python 2.5.  Older python versions do support
    # this suffix.
    if(WIN32 AND NOT CYGWIN)
      set_target_properties(${MY_KIT_NAME}Python PROPERTIES SUFFIX ".pyd")
    endif()

    # Make sure that no prefix is set on the library
    set_target_properties(${MY_KIT_NAME}Python PROPERTIES PREFIX "")

    install(TARGETS ${MY_KIT_NAME}Python
      RUNTIME DESTINATION ${MY_KIT_INSTALL_BIN_DIR} COMPONENT RuntimeLibraries
      LIBRARY DESTINATION ${MY_KIT_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${MY_KIT_INSTALL_LIB_DIR} COMPONENT Development
      )
  endif()

endmacro()

