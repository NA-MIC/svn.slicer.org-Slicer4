#ifndef __vtkUltrasoundExampleLibraryWin32Header_h
#define __vtkUltrasoundExampleLibraryWin32Header_h

//        #include <vtkUltrasoundModuleConfigure.h>

        #if defined(WIN32) && !defined(VTKSLICER_STATIC)
                #if defined(UltrasoundExampleGUILibrary_EXPORTS)
                        #define VTK_ULTRASOUNDEXAMPLEGUILIBRARY_EXPORT __declspec( dllexport ) 
                #else
                        #define VTK_ULTRASOUNDEXAMPLEGUILIBRARY_EXPORT __declspec( dllimport ) 
                #endif
        #else
                #define VTK_ULTRASOUNDEXAMPLEGUILIBRARY_EXPORT 
        #endif
#endif
