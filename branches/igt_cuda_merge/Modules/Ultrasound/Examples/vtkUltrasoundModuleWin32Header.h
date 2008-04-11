#ifndef __vtkUltrasoundLibraryWin32Header_h
#define __vtkUltrasoundLibraryWin32Header_h

//        #include <vtkUltrasoundModuleConfigure.h>

        #if defined(WIN32) && !defined(VTKSLICER_STATIC)
                #if defined(UltrasoundLibrary_EXPORTS)
                        #define VTK_ULTRASOUNDLIBRARY_EXPORT __declspec( dllexport ) 
                #else
                        #define VTK_ULTRASOUNDLIBRARY_EXPORT __declspec( dllimport ) 
                #endif
        #else
                #define VTK_ULTRASOUNDLIBRARY_EXPORT 
        #endif
#endif
