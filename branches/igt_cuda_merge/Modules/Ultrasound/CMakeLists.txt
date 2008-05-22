PROJECT(UltrasoundModule)

OPTION(ULTRASOUNDMODULE "Would you like to build Ultrasound Module support" ON)
if(ULTRASOUNDMODULE)

OPTION(PHILLIPS_ULTRASOUND_SCANNER "Would you like to compile in Support for the Phillips Ultrasound Scanners (proprietary support only)" ON)
if(PHILLIPS_ULTRASOUND_SCANNER)
   FIND_LIBRARY(PHILLIPS_ULTRASOUND_SCANNER_LIBRARY
      PipelineComDll
      C:\\dev\\lib
   )
   FIND_PATH(PHILLIPS_ULTRASOUND_SCANNER_INCLUDE_DIR
      StreamMgr.h
      C:\\dev\\include
   )

   SET(PHILLIPS_ULTRASOUND_SCANNER_SOURCES
      vtkPhilipsUltrasoundStreamSource.cxx
   )
   ADD_DEFINITIONS(-DPHILLIPS_ULTRASOUND_SCANNER_SUPPORT)
   MESSAGE("Adding Proprietary Support for the Phillips Ultrasound Scanner")
endif(PHILLIPS_ULTRASOUND_SCANNER)


SET(ASCENSION_MINIBIRD_SOURCES
    vtkMiniBirdInstrumentTracker.cxx
   )
FIND_PATH(ASCENSION_MINIBIRDS_INCLUDE_DIR
    Bird.h
    C:\\dev\\include)
FIND_LIBRARY(ASCENSION_MINIBIRDS_LIBRARY
    Bird
    C:\\dev\\lib
   )
# --------------------------------------------------------------------------
# Sources
  SET(UltrasoundModule_SRCS
  ## MAIN GUI
    vtkUltrasoundModuleGUI.cxx
    vtkUltrasoundModuleLogic.cxx
    vtkMRMLUltrasoundNode.cxx

  # GUI ELEMENTS
    vtkUltrasoundStreamerGUI.cxx
    
    vtkUltrasoundToolGUI.cxx
    
  # SCANNER CLASSES
    vtkUltrasoundStreamSource.cxx
    vtkUltrasoundScannerReader.cxx    
    ${PHILLIPS_ULTRASOUND_SCANNER_SOURCES}
    
  #Tracking
    ${ASCENSION_MINIBIRD_SOURCES}
)

#-----------------------------------------------
# Include dirs

INCLUDE_DIRECTORIES(
  ${UltrasoundModule_SOURCE_DIR}
  ${UltrasoundModule_BINARY_DIR}
  ${SlicerBase_SOURCE_DIR}
  ${SlicerBase_BINARY_DIR}
  ${SlicerBaseLogic_SOURCE_DIR}
  ${SlicerBaseLogic_BINARY_DIR}
  ${SlicerBaseGUI_SOURCE_DIR}
  ${SlicerBaseGUI_BINARY_DIR}
  ${vtkITK_SOURCE_DIR}
  ${vtkITK_BINARY_DIR}
  ${ITK_INCLUDE_DIRS}
  ${VTK_INCLUDE_DIRS}
  ${MRML_SOURCE_DIR}
  ${MRML_BINARY_DIR}
  ${PHILLIPS_ULTRASOUND_SCANNER_INCLUDE_DIR}
  ${ASCENSION_MINIBIRDS_INCLUDE_DIR}
  )


# --------------------------------------------------------------------------
# Wrapping

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE("${VTK_CMAKE_DIR}/vtkWrapTcl.cmake")
VTK_WRAP_TCL3(UltrasoundModule
         UltrasoundModule_TCL_SRCS 
         "${UltrasoundModule_SRCS}" 
         "")


# --------------------------------------------------------------------------
# Build the library

## Set the Libraries we link
SET(UltrasoundModule_LIBRARIES
  SlicerBaseLogic 
  SlicerBaseGUI 
  MRML
  vtkITK
  vtkCommonTCL
  vtkImagingTCL
  vtkFilteringTCL
  vtkIOTCL
  ${KWWidgets_LIBRARIES}
  ${PHILLIPS_ULTRASOUND_SCANNER_LIBRARY}
  ${ASCENSION_MINIBIRDS_LIBRARY}
 )


ADD_LIBRARY(UltrasoundModule
  ${UltrasoundModule_SRCS}
  ${UltrasoundModule_TCL_SRCS}
  )


TARGET_LINK_LIBRARIES(UltrasoundModule
  ${UltrasoundModule_LIBRARIES}
  )

IF(BUILD_SHARED_LIBS)
  INSTALL(TARGETS UltrasoundModule
      RUNTIME DESTINATION bin COMPONENT RuntimeLibraries 
      LIBRARY DESTINATION lib COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION lib COMPONENT Development)
ENDIF(BUILD_SHARED_LIBS)


# --------------------------------------------------------------------------
# Testing (requires some of the examples)

#IF(BUILD_TESTING)
#  ADD_SUBDIRECTORY(Testing)
#ENDIF(BUILD_TESTING)

##IF(BUILD_EXAMPLES)
  ADD_SUBDIRECTORY(Examples)
##ENDIF(BUILD_EXAMPLES)

CONFIGURE_FILE(
  ${UltrasoundModule_SOURCE_DIR}/vtkUltrasoundModuleConfigure.h.in 
  ${UltrasoundModule_BINARY_DIR}/vtkUltrasoundModuleConfigure.h
)
CONFIGURE_FILE(
  ${UltrasoundModule_SOURCE_DIR}/presets.xml
  ${UltrasoundModule_BINARY_DIR}/presets.xml
  COPYONLY
)
ENDIF(ULTRASOUNDMODULE)
