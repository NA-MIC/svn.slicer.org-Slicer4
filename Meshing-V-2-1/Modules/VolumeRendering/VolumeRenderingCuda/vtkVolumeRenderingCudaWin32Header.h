#ifndef __vtkVolumeRenderingCudaWin32Header_h
        #define __vtkVolumeRenderingCudaWin32Header_h

        #include <vtkVolumeRenderingCudaConfigure.h>

        #if defined(WIN32) && !defined(VTKSLICER_STATIC)
                #if defined(VolumeRenderingCuda_EXPORTS)
                        #define VTK_VOLUMERENDERINGCUDA_EXPORT __declspec( dllexport ) 
                #else
                        #define VTK_VOLUMERENDERINGCUDA_EXPORT __declspec( dllimport ) 
                #endif
        #else
                #define VTK_VOLUMERENDERINGCUDA_EXPORT 
        #endif
#endif
