#ifndef __vtkCudaModuleWin32Header_h
        #define __vtkCudaModuleWin32Header_h

        #include <vtkCudaSupportConfigure.h>

        #if defined(WIN32) && !defined(VTKSLICER_STATIC)
                #if defined(CUDASUPPORT_EXPORTS)
                        #define VTK_CUDASUPPORT_EXPORT __declspec( dllexport ) 
                #else
                        #define VTK_CUDASUPPORT_EXPORT __declspec( dllimport ) 
                #endif
        #else
                #define VTK_CUDASUPPORT_EXPORT 
        #endif
#endif
