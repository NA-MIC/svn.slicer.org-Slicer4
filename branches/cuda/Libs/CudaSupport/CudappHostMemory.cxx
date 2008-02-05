#include "CudappHostMemory.h"
#include "CudappBase.h"
#include "cuda_runtime_api.h"

#include <string.h>

CudappHostMemory::CudappHostMemory()
{
}

CudappHostMemory::~CudappHostMemory()
{
    this->Free();
}


void* CudappHostMemory::AllocateBytes(size_t count)
{
    this->Free();
    cudaError_t error = 
        cudaMallocHost(&this->MemPointer, count);
    this->Size = count;
    if (error != cudaSuccess)
        CudappBase::PrintError(error);

    return (void*) this->MemPointer;
}

void CudappHostMemory::Free()
{
    if (this->MemPointer != NULL)
    {
        cudaFreeHost(this->MemPointer);
        this->MemPointer = NULL;  
        this->Size = 0;
    }
}

void CudappHostMemory::PrintSelf(std::ostream&  os)
{
    this->CudappLocalMemory::PrintSelf(os);
}
