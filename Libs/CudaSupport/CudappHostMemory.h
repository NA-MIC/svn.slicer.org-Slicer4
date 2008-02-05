#ifndef VTKCUDAHOSTMEMORY_H_
#define VTKCUDAHOSTMEMORY_H_

#include "CudappLocalMemory.h"

//! Cuda Host Memory is page-locked host memory that is directly accesible from the cuda device
/**
* This memory is known to the graphics card and can be accessed very quickly.
* 
* @note to much host memory in page locked mode can reduce overall system performance.
*/
class VTK_CUDASUPPORT_EXPORT CudappHostMemory : public CudappLocalMemory
{
    vtkTypeRevisionMacro(CudappHostMemory, CudappLocalMemory);
public:
    static CudappHostMemory* New();

    virtual void* AllocateBytes(size_t count);
    virtual void Free();

    void PrintSelf(ostream& os, vtkIndent indent);

protected:
    CudappHostMemory();
    virtual ~CudappHostMemory();
    CudappHostMemory(const CudappHostMemory&);
    CudappHostMemory& operator=(const CudappHostMemory&);
};

#endif /*VTKCUDAHOSTMEMORY_H_*/
