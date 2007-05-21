
#ifndef __vtkBrpNavWin32Header_h
#define __vtkBrpNavWin32Header_h

#include <vtkBrpNavConfigure.h>

#if defined(WIN32) && !defined(VTKSLICER_STATIC)
#if defined(BrpNav_EXPORTS)
#define VTK_BRPNAV_EXPORT __declspec( dllexport ) 
#else
#define VTK_BRPNAV_EXPORT __declspec( dllimport ) 
#endif
#else
#define VTK_BRPNAV_EXPORT 
#endif
#endif
