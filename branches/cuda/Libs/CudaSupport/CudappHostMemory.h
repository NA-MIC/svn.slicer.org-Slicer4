#ifndef CUDAPPHOSTMEMORY_H_
#define CUDAPPHOSTMEMORY_H_

#include "CudappLocalMemory.h"

//! Cuda Host Memory is page-locked host memory that is directly accesible from the cuda device
/**
* This memory is known to the graphics card and can be accessed very quickly.
* 
* @note to much host memory in page locked mode can reduce overall system performance.
*/
class CUDA_SUPPORT_EXPORT CudappHostMemory : public CudappLocalMemory
{
public:
    CudappHostMemory();
    virtual ~CudappHostMemory();
    CudappHostMemory(const CudappHostMemory&);
    CudappHostMemory& operator=(const CudappHostMemory&);

    virtual void* AllocateBytes(size_t count);
    virtual void Free();

    void PrintSelf(std::ostream&  os);
};

#endif /*CUDAPPHOSTMEMORY_H_*/
