#
# qCTKBFDCheck.cmake - After this file is included into your main CMake script,
#                      HAVE_BFD will be defined if libbfd is available.
#

SET(qCTKWidgets_BFD_LIBRARIES)
UNSET(qCTKWidgets_HAVE_BFD)

IF(NOT WIN32)
  INCLUDE(CheckIncludeFile)
  CHECK_INCLUDE_FILE(bfd.h HAVE_BFD_HEADER)

  IF(HAVE_BFD_HEADER)
    # make sure we can build with libbfd
    #MESSAGE(STATUS "Checking libbfd")
    TRY_COMPILE(qCTKWidgets_HAVE_BFD
      ${CMAKE_CURRENT_BINARY_DIR}/CMake/TestBFD
      ${CMAKE_CURRENT_SOURCE_DIR}/CMake/TestBFD
      TestBFD
      CMAKE_FLAGS
      -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      OUTPUT_VARIABLE OUTPUT)
    #MESSAGE(${OUTPUT})

    IF(qCTKWidgets_HAVE_BFD)
      SET(qCTKWidgets_BFD_LIBRARIES bfd iberty)
      MESSAGE(STATUS "qCTKWidgets: libbfd is available")
    ELSE(qCTKWidgets_HAVE_BFD)
      MESSAGE(STATUS "qCTKWidgets: libbfd is *NOT* available")
    ENDIF(qCTKWidgets_HAVE_BFD)

  ENDIF(HAVE_BFD_HEADER)
ENDIF(NOT WIN32)
