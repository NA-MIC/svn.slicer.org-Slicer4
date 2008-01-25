#include "vtkCudaLocalMemory.h"

#include "vtkCudaBase.h"
#include "vtkObjectFactory.h"

#include <string.h>
#include "cuda_runtime_api.h"

vtkCxxRevisionMacro(vtkCudaLocalMemory, "$Revision 1.0 $");
vtkStandardNewMacro(vtkCudaLocalMemory);

vtkCudaLocalMemory::vtkCudaLocalMemory()
{
  this->Location = vtkCudaMemoryBase::MemoryOnHost;
}

vtkCudaLocalMemory::~vtkCudaLocalMemory()
{
    this->Free();
}

void* vtkCudaLocalMemory::AllocateBytes(size_t count)
{
    this->Free();
    this->MemPointer = malloc(count);
    this->Size = count;
    if (this->MemPointer == NULL)
        ;

    return (void*)this->MemPointer;
}

void vtkCudaLocalMemory::Free()
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
void vtkCudaLocalMemory::MemSet(int value)
{
    memset(this->MemPointer, value, Size);
}


bool vtkCudaLocalMemory::CopyTo(void* dst, size_t byte_count, size_t offset, MemoryLocation dst_loc)
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

bool vtkCudaLocalMemory::CopyFrom(void* src, size_t byte_count, size_t offset, MemoryLocation src_loc)
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


void vtkCudaLocalMemory::PrintSelf(ostream& os, vtkIndent indent)
{
    this->Superclass::PrintSelf(os, indent);
}
