PROJECT(FMTractography)

CMAKE_MINIMUM_REQUIRED(VERSION 2.4)
#find_package(Slicer3 REQUIRED)
#include(${Slicer3_USE_FILE})
#slicer3_set_default_install_prefix_for_external_projects()

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

set(CMAKE_CXX_FLAGS "-w -fpermissive -static -static-libgcc -fexceptions")
set(CMAKE_EXE_LINKER_FLAGS "-Wl,-dylib_file,/System/Library/Frameworks/OpenGL.framework/Versions/A/Libraries/libGL.dylib:/System/Library/Frameworks/OpenGL.framework/Versions/A/Libraries/libGL.dylib")

set(CLP FMTractography)
set(${CLP}_SOURCE ${CLP}.xml)
generateclp(${CLP}_SOURCE ${CLP}.xml)
ADD_EXECUTABLE(${CLP} ${${CLP}_SOURCE} FMTractography.cpp)
slicer3_set_plugins_output_path(${CLP})
target_link_libraries(${CLP} ${ITK_LIBRARIES} ${VTK_LIBRARIES} ITKCommon ITKIO vtkHybrid vtkParallel)
slicer3_install_plugins(${CLP})
