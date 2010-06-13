# The Numpy external project 

set(numpy_binary "${CMAKE_CURRENT_BINARY_DIR}/NUMPY/")

# to configure numpy we run a cmake -P script
# the script will create a site.cfg file
# then run python setup.py config to verify setup
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/NUMPY_configure_step.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/NUMPY_configure_step.cmake @ONLY)
# to build numpy we also run a cmake -P script.
# the script will set LD_LIBRARY_PATH so that 
# python can run after it is built on linux
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/NUMPY_make_step.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/NUMPY_make_step.cmake @ONLY)

# create an external project to download numpy,
# and configure and build it
ExternalProject_Add(NUMPY
  DOWNLOAD_DIR ${CMAKE_CURRENT_BINARY_DIR}
  #URL "http://iweb.dl.sourceforge.net/project/numpy/NumPy/1.4.1/numpy-1.4.1.tar.gz"
  SVN_REPOSITORY http://svn.scipy.org/svn/numpy/trunk
  SVN_REVISION -r "8454"
  SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/NUMPY
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/NUMPY
  CONFIGURE_COMMAND ${CMAKE_COMMAND}
    -P ${CMAKE_CURRENT_BINARY_DIR}/NUMPY_configure_step.cmake
  BUILD_COMMAND ${CMAKE_COMMAND}
    -P ${CMAKE_CURRENT_BINARY_DIR}/NUMPY_make_step.cmake
  UPDATE_COMMAND ""
  INSTALL_COMMAND ""
  DEPENDS ${numpy_DEPENDENCIES}
  )
