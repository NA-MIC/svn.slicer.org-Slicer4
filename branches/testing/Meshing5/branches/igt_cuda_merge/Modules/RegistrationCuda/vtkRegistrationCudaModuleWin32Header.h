#ifndef __vtkRegistrationCudaModuleWin32Header_h
        #define __vtkRegistrationCudaModuleWin32Header_h

        #include <vtkRegistrationCudaModuleConfigure.h>

        #if defined(WIN32) && !defined(VTKSLICER_STATIC)
                #if defined(RegistrationCudaModule_EXPORTS)
                        #define VTK_REGISTRATIONCUDAMODULE_EXPORT __declspec( dllexport ) 
                #else
                        #define VTK_REGISTRATIONCUDAMODULE_EXPORT __declspec( dllimport ) 
                #endif
        #else
                #define VTK_REGISTRATIONCUDAMODULE_EXPORT 
        #endif
#endif
