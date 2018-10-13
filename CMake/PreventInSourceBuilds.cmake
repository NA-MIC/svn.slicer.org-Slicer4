# Adapted from ITKv4/CMake/PreventInSourceBuilds.cmake
#
# This function will prevent in-source builds
function(AssureOutOfSourceBuilds)
  # make sure the user doesn't play dirty with symlinks
  get_filename_component(srcdir "${CMAKE_SOURCE_DIR}" REALPATH)
  get_filename_component(bindir "${CMAKE_BINARY_DIR}" REALPATH)

  # disallow in-source builds
  if("${srcdir}" STREQUAL "${bindir}")
    message("######################################################")
    message("# ${PROJECT_NAME} should not be configured & built in the ${PROJECT_NAME} source directory")
    message("# You must run cmake in a build directory.")
    message("# For example:")
    message("# mkdir ${PROJECT_NAME}-sandbox ; cd ${PROJECT_NAME}-sandbox")
    message("# git clone git://github.com/${PROJECT_NAME}/${PROJECT_NAME}.git # or download & unpack the source tarball")
    message("# mkdir ${PROJECT_NAME}-SuperBuild")
    message("# this will create the following directory structure")
    message("#")
    message("# ${PROJECT_NAME}-sandbox")
    message("#  +--${PROJECT_NAME}")
    message("#  +--${PROJECT_NAME}-SuperBuild")
    message("#")
    message("# Then you can proceed to configure and build")
    message("# by using the following commands")
    message("#")
    message("# cd ${PROJECT_NAME}-SuperBuild")
    message("# cmake ../${PROJECT_NAME} # or ccmake, or cmake-gui ")
    message("# make")
    message("#")
    message("# NOTE: Given that you already tried to make an in-source build")
    message("#       CMake have already created several files & directories")
    message("#       in your source tree. run 'git status' to find them and")
    message("#       remove them by doing:")
    message("#")
    message("#       cd ${PROJECT_NAME}-sandbox/${PROJECT_NAME}")
    message("#       git clean -n -d")
    message("#       git clean -f -d")
    message("#       git checkout --")
    message("#")
    message("######################################################")
    message(FATAL_ERROR "Quitting configuration")
  endif()
endfunction()

AssureOutOfSourceBuilds()
