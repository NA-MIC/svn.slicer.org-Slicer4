#ifndef __vtkCudaModuleWin32Header_h
        #define __vtkCudaModuleWin32Header_h

        #include <vtkCudaModuleConfigure.h>

        #if defined(WIN32) && !defined(VTKSLICER_STATIC)
                #if defined(CUDAMODULE_EXPORTS)
                        #define VTK_CUDAMODULE_EXPORT __declspec( dllexport ) 
                #else
                        #define VTK_CUDAMODULE_EXPORT __declspec( dllimport ) 
                #endif
        #else
                #define VTK_CUDAMODULE_EXPORT 
        #endif
#endif
