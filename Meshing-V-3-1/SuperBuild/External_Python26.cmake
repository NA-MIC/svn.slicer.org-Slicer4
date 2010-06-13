#-----------------------------------------------------------------------------
set(proj python)
set(python_base ${CMAKE_CURRENT_BINARY_DIR}/${proj})
set(python_build ${CMAKE_CURRENT_BINARY_DIR}/${proj}-build)
set(python_SVN_REPOSITORY "http://svn.python.org/projects/python/branches/release26-maint")
set(python_SVN_REVISION -r 81659)

if(WIN32)

  set(python_sln ${CMAKE_BINARY_DIR}/${proj}-build/PCbuild/pcbuild.sln)
  string(REPLACE "/" "\\" python_sln ${python_sln})

  get_filename_component(python_base ${python_sln} PATH)
  get_filename_component(python_home ${python_base} PATH)

  # point the tkinter build file to the slicer tcl-build
  set(python_PATCH_COMMAND)
  if(Slicer3_USE_KWWIDGETS)
    set(python_tkinter ${python_base}/pyproject.vsprops)
    string(REPLACE "/" "\\" python_tkinter ${python_tkinter})

    set(script ${CMAKE_CURRENT_SOURCE_DIR}/../CMake/StringFindReplace.cmake)
    set(out ${python_tkinter})
    set(in ${python_tkinter})

    set(python_PATCH_COMMAND 
      ${CMAKE_COMMAND} -Din=${in} -Dout=${out} -Dfind=tcltk\" -Dreplace=tcl-build\" -P ${script})
  endif()

  ExternalProject_Add(${proj}
    DEPENDS ${python_DEPENDENCIES}
    SVN_REPOSITORY ${python_SVN_REPOSITORY}
    SVN_REVISION ${python_SVN_REVISION}
    SOURCE_DIR python-build
    UPDATE_COMMAND ""
    PATCH_COMMAND ${python_PATCH_COMMAND}
    CONFIGURE_COMMAND ${CMAKE_BUILD_TOOL} ${python_sln} /Upgrade
    BUILD_COMMAND ${CMAKE_BUILD_TOOL} ${python_sln} /build Release /project select
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND ""
  )

  if(Slicer3_USE_KWWIDGETS)
    # this must match the version of tcl we are building for slicer.
    ExternalProject_Add_Step(${proj} Patch_tcltk_version
      COMMAND ${CMAKE_COMMAND} -Din=${in} -Dout=${out} -Dfind=85 -Dreplace=84 -P ${script}
      DEPENDEES configure
      DEPENDERS build
      )
  endif()

  ExternalProject_Add_Step(${proj} Build_make_versioninfo
    COMMAND ${CMAKE_BUILD_TOOL} ${python_sln} /build Release /project make_versioninfo
    DEPENDEES configure
    )

  ExternalProject_Add_Step(${proj} Build_make_buildinfo
    COMMAND ${CMAKE_BUILD_TOOL} ${python_sln} /build Release /project make_buildinfo
    DEPENDEES Build_make_versioninfo
    )

  ExternalProject_Add_Step(${proj} Build_kill_python
    COMMAND ${CMAKE_BUILD_TOOL} ${python_sln} /build Release /project kill_python
    DEPENDEES Build_kill_python
    )

  ExternalProject_Add_Step(${proj} Build_w9xpopen
    COMMAND ${CMAKE_BUILD_TOOL} ${python_sln} /build Release /project w9xpopen
    DEPENDEES Build_kill_python
    )

  ExternalProject_Add_Step(${proj} Build_pythoncore
    COMMAND ${CMAKE_BUILD_TOOL} ${python_sln} /build Release /project pythoncore
    DEPENDEES Build_w9xpopen
    )

  ExternalProject_Add_Step(${proj} Build__socket
    COMMAND ${CMAKE_BUILD_TOOL} ${python_sln} /build Release /project _socket
    DEPENDEES Build_pythoncore
    )

  if(Slicer3_USE_KWWIDGETS)
    ExternalProject_Add_Step(${proj} Build__tkinter
      COMMAND ${CMAKE_BUILD_TOOL} ${python_sln} /build Release /project _tkinter
      DEPENDEES Build__socket
      )
  endif()

  ExternalProject_Add_Step(${proj} Build__testcapi
    COMMAND ${CMAKE_BUILD_TOOL} ${python_sln} /build Release /project _testcapi
    DEPENDEES Build_pythoncore
    )

  ExternalProject_Add_Step(${proj} Build__msi
    COMMAND ${CMAKE_BUILD_TOOL} ${python_sln} /build Release /project _msi
    DEPENDEES Build__testcapi
    )

  ExternalProject_Add_Step(${proj} Build__elementtree
    COMMAND ${CMAKE_BUILD_TOOL} ${python_sln} /build Release /project _elementtree
    DEPENDEES Build__msi
    )

  ExternalProject_Add_Step(${proj} Build__ctypes_test
    COMMAND ${CMAKE_BUILD_TOOL} ${python_sln} /build Release /project _ctypes_test
    DEPENDEES Build__elementtree
    )

  ExternalProject_Add_Step(${proj} Build__ctypes
    COMMAND ${CMAKE_BUILD_TOOL} ${python_sln} /build Release /project _ctypes
    DEPENDEES python_sln
    )

  ExternalProject_Add_Step(${proj} Build_winsound
    COMMAND ${CMAKE_BUILD_TOOL} ${python_sln} /build Release /project winsound
    DEPENDEES Build__ctypes
    )

  ExternalProject_Add_Step(${proj} Build_pyexpat
    COMMAND ${CMAKE_BUILD_TOOL} ${python_sln} /build Release /project pyexpat
    DEPENDEES Build_winsound
    )

  ExternalProject_Add_Step(${proj} Build_pythonw
    COMMAND ${CMAKE_BUILD_TOOL} ${python_sln} /build Release /project pythonw
    DEPENDEES Build_pyexpat
    )

  ExternalProject_Add_Step(${proj} Build__multiprocessing
    COMMAND ${CMAKE_BUILD_TOOL} ${python_sln} /build Release /project _multiprocessing
    DEPENDEES Build_pythonw
    )
    
  ExternalProject_Add_Step(${proj} Build_python
    COMMAND ${CMAKE_BUILD_TOOL} ${python_sln} /build Release /project python
    DEPENDEES _multiprocessing
    )

  ExternalProject_Add_Step(${proj} CopyPythonLib
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/python-build/PCbuild/python26.lib ${CMAKE_BINARY_DIR}/python-build/Lib/python26.lib
    DEPENDEES install
    )
  ExternalProject_Add_Step(${proj} Copy_socketPyd
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/python-build/PCbuild/_socket.pyd ${CMAKE_BINARY_DIR}/python-build/Lib/_socket.pyd
    DEPENDEES install
    )
  ExternalProject_Add_Step(${proj} Copy_ctypesPyd
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/python-build/PCbuild/_ctypes.pyd ${CMAKE_BINARY_DIR}/python-build/Lib/_ctypes.pyd
    DEPENDEES install
    )

  ExternalProject_Add_Step(${proj} CopyPythonDll
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/python-build/PCbuild/python26.dll ${CMAKE_BINARY_DIR}/Slicer3-build/bin/${CMAKE_CFG_INTDIR}/python26.dll
    DEPENDEES install
    )
    
elseif(UNIX)
  set(python_BUILD_IN_SOURCE 1)
  
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/python_configure_step.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/python_configure_step.cmake
    @ONLY)
    
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/python_make_step.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/python_make_step.cmake
    @ONLY)
    
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/python_install_step.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/python_install_step.cmake
    @ONLY)

  set(python_SOURCE_DIR python)
  set(python_CONFIGURE_COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/python_configure_step.cmake)
  set(python_BUILD_COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/python_make_step.cmake)
  set(python_INSTALL_COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/python_install_step.cmake)
  
  ExternalProject_Add(${proj}
    SVN_REPOSITORY ${python_SVN_REPOSITORY}
    SVN_REVISION ${python_SVN_REVISION}
    DEPENDS ${python_DEPENDENCIES}
    SOURCE_DIR ${python_SOURCE_DIR}
    BUILD_IN_SOURCE ${python_BUILD_IN_SOURCE}
    CONFIGURE_COMMAND ${python_CONFIGURE_COMMAND}
    BUILD_COMMAND ${python_BUILD_COMMAND}
    UPDATE_COMMAND ""
    INSTALL_COMMAND ${python_INSTALL_COMMAND}
    )

endif()

#-----------------------------------------------------------------------------
# Set slicer_PYTHON_INCLUDE and slicer_PYTHON_LIBRARY variables
#

set(slicer_PYTHON_INCLUDE)
set(slicer_PYTHON_LIBRARY)
set(slicer_PYTHON_EXECUTABLE)

if(WIN32)
  set(slicer_PYTHON_INCLUDE ${CMAKE_BINARY_DIR}/Python-build/Include)
  set(slicer_PYTHON_LIBRARY ${CMAKE_BINARY_DIR}/Python-build/PCbuild/python26.lib)
  set(slicer_PYTHON_EXECUTABLE ${CMAKE_BINARY_DIR}/Python-build/PCbuild/python.exe)
elseif(APPLE)
  set(slicer_PYTHON_INCLUDE ${CMAKE_BINARY_DIR}/python-build/include/python2.6)
  set(slicer_PYTHON_LIBRARY ${CMAKE_BINARY_DIR}/python-build/lib/libpython2.6.dylib)
  set(slicer_PYTHON_EXECUTABLE ${CMAKE_BINARY_DIR}/python-build/bin/python)
else()
  set(slicer_PYTHON_INCLUDE ${CMAKE_BINARY_DIR}/python-build/include/python2.6)
  set(slicer_PYTHON_LIBRARY ${CMAKE_BINARY_DIR}/python-build/lib/libpython2.6.so)
  set(slicer_PYTHON_EXECUTABLE ${CMAKE_BINARY_DIR}/python-build/bin/python)
endif()

