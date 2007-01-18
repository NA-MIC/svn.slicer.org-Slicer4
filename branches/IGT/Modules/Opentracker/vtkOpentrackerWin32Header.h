
#ifndef __vtkOpentrackerWin32Header_h
#define __vtkOpentrackerWin32Header_h

#include <vtkOpentrackerConfigure.h>

#if defined(WIN32) && !defined(VTKSLICER_STATIC)
#if defined(Opentracker_EXPORTS)
#define VTK_OPENTRACKER_EXPORT __declspec( dllexport ) 
#else
#define VTK_OPENTRACKER_EXPORT __declspec( dllimport ) 
#endif
#else
#define VTK_OPENTRACKER_EXPORT 
#endif
#endif
