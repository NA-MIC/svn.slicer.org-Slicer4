#ifndef __CudappModuleWin32Header_h
        #define __CudappModuleWin32Header_h

        #include <CudappSupportConfigure.h>

        #if defined(WIN32) && !defined(VTKSLICER_STATIC)
                #if defined(CudaSupport_EXPORTS)
                        #define VTK_CUDASUPPORT_EXPORT __declspec( dllexport ) 
                #else
                        #define VTK_CUDASUPPORT_EXPORT __declspec( dllimport ) 
                #endif
        #else
                #define VTK_CUDASUPPORT_EXPORT 
        #endif
#endif
