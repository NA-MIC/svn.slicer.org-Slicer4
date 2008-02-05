#include "CudappLocalMemory.h"

#include "CudappBase.h"

#include <string.h>
#include "cuda_runtime_api.h"

CudappLocalMemory::CudappLocalMemory()
{
  this->Location = CudappMemoryBase::MemoryOnHost;
}

CudappLocalMemory::~CudappLocalMemory()
{
    this->Free();
}

void* CudappLocalMemory::AllocateBytes(size_t count)
{
    this->Free();
    this->MemPointer = malloc(count);
    this->Size = count;
    if (this->MemPointer == NULL)
        ;

    return (void*)this->MemPointer;
}

void CudappLocalMemory::Free()
{
    if (this->MemPointer != NULL)
    {
        free (this->MemPointer);
        this->MemPointer = NULL;  
        this->Size = 0;
    }
}

/**
* @brief host implementation of the MemorySetter Value
*/
void CudappLocalMemory::MemSet(int value)
{
    memset(this->MemPointer, value, Size);
}


bool CudappLocalMemory::CopyTo(void* dst, size_t byte_count, size_t offset, MemoryLocation dst_loc)
{
  if(cudaMemcpy(dst, 
        this->GetMemPointer(), //HACK  + offset,
        byte_count,
        (dst_loc == MemoryOnHost) ? cudaMemcpyHostToHost : cudaMemcpyHostToDevice
        ) == cudaSuccess)
        return true;
    else 
        return false;
}

bool CudappLocalMemory::CopyFrom(void* src, size_t byte_count, size_t offset, MemoryLocation src_loc)
{
    if(cudaMemcpy(this->GetMemPointer(), //HACK  + offset, 
        src,
        byte_count,
        (src_loc == MemoryOnHost) ? cudaMemcpyHostToHost : cudaMemcpyDeviceToHost
        ) == cudaSuccess)
        return true;
    else 
        return false;
}


void CudappLocalMemory::PrintSelf(std::ostream&  os)
{
    this->CudappMemory::PrintSelf(os);
}
