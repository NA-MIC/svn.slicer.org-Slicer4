
  message ( STATUS "Configuration of Python modules, python is ${PYTHON_EXECUTABLE} path is $ENV{PATH}" )

  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/FilteredTractography_make_step.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/FilteredTractography_make_step.cmake @ONLY )

  set ( FilteredTractography_source ${CMAKE_CURRENT_SOURCE_DIR} )

  # To build the shared library, we run the BuildFilteredTractography.py script, but the weave tools must be installed
  add_custom_command ( OUTPUT ${CMAKE_BINARY_DIR}/${Slicer3_INSTALL_PLUGINS_BIN_DIR}/filtered_ext.so
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/FilteredTractography_make_step.cmake 
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/FilteredTractography.py
    WORKING_DIRECTORY "${Slicer3_BINARY_DIR}/${Slicer3_INSTALL_PLUGINS_BIN_DIR}"
    COMMENT "Building shared library for FilteredTractography"
    )

  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/FilteredTractography.py
    ${Slicer3_BINARY_DIR}/${Slicer3_INSTALL_PLUGINS_BIN_DIR}/FilteredTractography.py
    COPYONLY )

  install(
    FILES FilteredTractography/FilteredTractography.py
    DESTINATION ${Slicer3_INSTALL_PLUGINS_BIN_DIR}/FilteredTractography.py
    )

  add_custom_target ( filtered1t_Library ALL DEPENDS ${CMAKE_BINARY_DIR}/${Slicer3_INSTALL_PLUGINS_BIN_DIR}/filtered_ext.so )

