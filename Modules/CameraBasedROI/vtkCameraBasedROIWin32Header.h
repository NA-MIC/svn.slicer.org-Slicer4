#ifndef __vtkCameraBasedROIWin32Header_h
#define __vtkCameraBasedROIWin32Header_h

#include <vtkCameraBasedROIConfigure.h>

#if defined(WIN32) && !defined(VTKSLICER_STATIC)
#if defined(CameraBasedROI_EXPORTS)
#define VTK_CAMERABASEDROI_EXPORT __declspec( dllexport ) 
#else
#define VTK_CAMERABASEDROI_EXPORT __declspec( dllimport ) 
#endif
#else
#define VTK_CAMERABASEDROI_EXPORT 
#endif

#endif
