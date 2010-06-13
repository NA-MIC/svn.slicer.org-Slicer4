#-----------------------------------------------------------------------------
# Get and build CTK
set(ctk_source ${CMAKE_BINARY_DIR}/CTK)
if (Slicer3_USE_QT)
  ExternalProject_Add(CTK
    DEPENDS vtk
    GIT_REPOSITORY "git://github.com/pieper/CTK.git"
    SOURCE_DIR ${ctk_source}
    BINARY_DIR CTK-build
    CMAKE_GENERATOR ${gen}
    CMAKE_ARGS
      -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
      -DCMAKE_C_FLAGS:STRING=${CMAKE_C_FLAGS}
      -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
      -DBUILD_TESTING:BOOL=OFF
      -DQT_QMAKE_EXECUTABLE:FILEPATH=${QT_QMAKE_EXECUTABLE}
      -DVTK_DIR:PATH=${CMAKE_BINARY_DIR}/VTK-build
      -DPYTHON_LIBRARY:FILEPATH=${slicer_PYTHON_LIBRARY}
      -DPYTHON_INCLUDE_DIR:PATH=${slicer_PYTHON_INCLUDE}
      -DCTK_LIB_Widgets:BOOL=ON
      -DCTK_LIB_Visualization/VTK/Widgets:BOOL=ON
      -DCTK_LIB_Scripting/Python/Widgets:BOOL=${Slicer3_USE_PYTHONQT}
      -DCTK_LIB_PluginFramework:BOOL=OFF
      -DCTK_PLUGIN_org.commontk.eventbus:BOOL=OFF
    INSTALL_COMMAND ""
    
    )
endif()
