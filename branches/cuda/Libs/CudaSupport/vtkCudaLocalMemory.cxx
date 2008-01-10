#include "vtkCudaLocalMemory.h"

#include "vtkCudaBase.h"
#include "vtkObjectFactory.h"

#include <string.h>
#include "cuda_runtime_api.h"

vtkCxxRevisionMacro(vtkCudaLocalMemory, "$Revision 1.0 $");
vtkStandardNewMacro(vtkCudaLocalMemory);

vtkCudaLocalMemory::vtkCudaLocalMemory()
{
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

bool vtkCudaLocalMemory::CopyTo(vtkCudaMemory* other)
{
    other->AllocateBytes(this->GetSize());
    if(cudaMemcpy(other->GetMemPointer(), 
        this->GetMemPointer(),
        this->GetSize(),
        cudaMemcpyHostToDevice
        ) == cudaSuccess)
        return true;
    else 
        return false;
}

bool vtkCudaLocalMemory::CopyTo(vtkCudaLocalMemory* other)
{
    other->AllocateBytes(this->GetSize());
    if(cudaMemcpy(other->GetMemPointer(), 
        this->GetMemPointer(),
        this->GetSize(),
        cudaMemcpyHostToHost
        ) == cudaSuccess)
        return true;
    else 
        return false;
}

void vtkCudaLocalMemory::PrintSelf(ostream& os, vtkIndent indent)
{
    this->Superclass::PrintSelf(os, indent);
}
