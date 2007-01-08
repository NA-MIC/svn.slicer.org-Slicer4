
#ifndef __vtkNeurosurgeryWin32Header_h
#define __vtkNeurosurgeryWin32Header_h

#include <vtkNeurosurgeryConfigure.h>

#if defined(WIN32) && !defined(VTKSLICER_STATIC)
#if defined(Neurosurgery_EXPORTS)
#define VTK_NEUROSURGERY_EXPORT __declspec( dllexport ) 
#else
#define VTK_NEUROSURGERY_EXPORT __declspec( dllimport ) 
#endif
#else
#define VTK_NEUROSURGERY_EXPORT 
#endif
#endif
