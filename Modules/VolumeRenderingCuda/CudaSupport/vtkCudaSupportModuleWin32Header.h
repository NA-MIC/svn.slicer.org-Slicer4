#ifndef __vtkCudaModuleWin32Header_h
        #define __vtkCudaModuleWin32Header_h

        #include <vtkCudaSupportModuleConfigure.h>

        #if defined(WIN32) && !defined(VTKSLICER_STATIC)
                #if defined(CUDASUPPORTMODULE_EXPORTS)
                        #define VTK_CUDASUPPORTMODULE_EXPORT __declspec( dllexport ) 
                #else
                        #define VTK_CUDASUPPORTMODULE_EXPORT __declspec( dllimport ) 
                #endif
        #else
                #define VTK_CUDASUPPORTMODULE_EXPORT 
        #endif
#endif
