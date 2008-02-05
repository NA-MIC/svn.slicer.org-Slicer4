#ifndef VTKCUDABASE_H_
#define VTKCUDABASE_H_

#include "vtkObject.h"
#include "driver_types.h"
#include "CudappSupportModule.h"

/// THIS IS A STATIC CLASS USED FOR BASIC CUDA FUNCTIONALITY!!
class VTK_CUDASUPPORT_EXPORT CudappBase : public vtkObject
{
    vtkTypeRevisionMacro(CudappBase, vtkObject);
public:
    //BTX
    typedef enum {
        Success,
        NotReadyError,
        InvalidValueError,
    } State;
    //ETX

    static CudappBase* New();

    static cudaError_t GetLastError();
    static const char* GetLastErrorString();
    static const char* GetErrorString(cudaError_t error);
    static void PrintError(cudaError_t error);

private:
    virtual ~CudappBase();
    CudappBase();
};

#endif /*VTKCUDABASE_H_*/
