#ifndef __vtkUltrasoundModuleWin32Header_h
#define __vtkUltrasoundModuleWin32Header_h

        #include <vtkUltrasoundModuleConfigure.h>

        #if defined(WIN32) && !defined(VTKSLICER_STATIC)
                #if defined(UltrasoundModule_EXPORTS)
                        #define VTK_ULTRASOUNDMODULE_EXPORT __declspec( dllexport ) 
                #else
                        #define VTK_ULTRASOUNDMODULE_EXPORT __declspec( dllimport ) 
                #endif
        #else
                #define VTK_ULTRASOUNDMODULE_EXPORT 
        #endif
#endif
