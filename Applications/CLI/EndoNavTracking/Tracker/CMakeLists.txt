project(EndoNavTracker) 

find_package(GenerateCLP REQUIRED)
if(GenerateCLP_FOUND)
  include(${GenerateCLP_USE_FILE})
endif(GenerateCLP_FOUND)

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})


set (CLP EndoNavTracker)
set ( ${CLP}_SOURCE 
    ${CLP}.cxx
    vtkTrackingSystem.cxx
    vtkMiniBirdTrackingSystem.cxx
    vtkMicroBirdTrackingSystem.cxx 
    vtkTrackedVideo.cxx
    vtkTrackedVideoData.cxx
    vtkTrackFileData.cxx
    #vtkWin32TrackedVideoSource.cxx
)
generateclp(${CLP}_SOURCE ${CLP}.xml ${Slicer3_SOURCE_DIR}/Resources/NAMICLogo.h)

# Add additional directories to look in for libraries
LINK_DIRECTORIES(
${CMAKE_CURRENT_SOURCE_DIR}/Lib
)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${VTK_INCLUDE_DIRS}
  ${OpenIGTLink_INCLUDE_DIRS}
  )


add_executable(${CLP} ${${CLP}_SOURCE})
slicer3_set_plugins_output_path(${CLP})


# To add a list of additional libraries you need to link with, uncomment
# the following line.  The library names should omit the "lib" and any
# trailing suffix.

target_link_libraries (${CLP}
  ${VTK_LIBRARIES}
  PCIBird3 Bird
  vtkIO vtkGraphics vtkImaging vtkFiltering vtkCommon vtkHybrid vtkRendering
  ${OpenIGTLink_LIBRARIES}
  ModuleDescriptionParser
  # MRML
  )

if (Slicer3_SOURCE_DIR)
  # install each target in the production area (where it would appear in an 
  # installation) and install each target in the developer area (for running 
  # from a build)
  set(TARGETS        
    EndoNavTracker
    )
  slicer3_install_plugins(${TARGETS})

endif (Slicer3_SOURCE_DIR)

file(GLOB TRACKERLIBS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/Lib/" "*.dll")
slicer3_install_plugins(${TRACKERLIBS})

#foreach(file ${TRACKERLIBS})
#  configure_file(
#    ${CMAKE_CURRENT_SOURCE_DIR}/${file}
#    ${CMAKE_BINARY_DIR}/${Slicer3_INSTALL_PLUGIN_BIN_DIR}/${file}
#    COPYONLY
#    )
#endforeach(file)

#install(
#  FILES ${TRACKERLIBS}
#  DESTINATION ${Slicer3_INSTALL_PLUGIN_BIN_DIR}
#  )
