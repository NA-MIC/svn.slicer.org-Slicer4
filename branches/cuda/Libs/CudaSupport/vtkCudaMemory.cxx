#include "vtkCudaMemory.h"

#include "cuda_runtime_api.h"
#include "vtkCudaBase.h"

#include "vtkCudaLocalMemory.h"
#include "vtkCudaHostMemory.h"
#include "vtkCudaMemoryArray.h"

#include "vtkObjectFactory.h"
#include "vtkImageData.h"

vtkCxxRevisionMacro(vtkCudaMemory, "$Revision: 1.0 $");
vtkStandardNewMacro(vtkCudaMemory);

vtkCudaMemory::vtkCudaMemory()
{
    this->MemPointer = NULL;
    this->Size = 0;
}

vtkCudaMemory::~vtkCudaMemory()
{
    // so the virtual function call will not be false.
    // each subclass must call free by its own and set MemPointer to NULL in its Destructor!
    if (this->MemPointer != NULL)
        this->Free();
}

void vtkCudaMemory::Free()
{
    if (this->MemPointer != NULL)
    {
        cudaFree(this->MemPointer);  
        this->MemPointer = NULL;
        this->Size = 0;
    }
}

void* vtkCudaMemory::AllocateBytes(size_t byte_count)
{
    // do nothing in case we already allocated the desired size.
    if (this->GetSize() == byte_count)
        return this->MemPointer;

    this->Free();
    cudaError_t error = 
        cudaMalloc(&this->MemPointer, byte_count);
    this->Size = byte_count;
    if (error != cudaSuccess)
        vtkCudaBase::PrintError(error);

    return (void*) this->MemPointer;
}

void vtkCudaMemory::MemSet(int value)
{
    cudaMemset(this->MemPointer, value, this->Size);
}

bool vtkCudaMemory::CopyFrom(vtkImageData* data)
{
    this->AllocateBytes(data->GetActualMemorySize()*1024);

    if(cudaMemcpy(this->GetMemPointer(),
        data->GetScalarPointer(),
        this->GetSize(),
        cudaMemcpyHostToDevice
        ) == cudaSuccess)
        return true;
    else 
        return false;
}

bool vtkCudaMemory::CopyTo(vtkImageData* data)
{
    if(data->GetActualMemorySize() * 1024 < this->GetSize())
    {
        vtkErrorMacro("The vtkImageData has to little Memory to store memory inside");
        return false;
    }
    // we cannot say the XYZ extent so we just write into the memory area
    if(cudaMemcpy(data->GetScalarPointer(),
        this->GetMemPointer(),
        this->GetSize(),
        cudaMemcpyDeviceToHost
        ) == cudaSuccess)
        return true;
    else 
        return false;
}

bool vtkCudaMemory::CopyTo(vtkCudaMemory* other)
{
    other->AllocateBytes(this->GetSize());
    if(cudaMemcpy(other->GetMemPointer(), 
        this->GetMemPointer(),
        this->GetSize(),
        cudaMemcpyDeviceToDevice
        ) == cudaSuccess)
        return true;
    else 
        return false;
}

bool vtkCudaMemory::CopyTo(vtkCudaLocalMemory* other)
{
    other->AllocateBytes(this->GetSize());
    if(cudaMemcpy(other->GetMemPointer(), 
        this->GetMemPointer(),
        this->GetSize(),
        cudaMemcpyDeviceToHost
        ) == cudaSuccess)
        return true;
    else 
        return false;
}

bool vtkCudaMemory::CopyTo(vtkCudaMemoryArray* other)
{
    if (other->GetSize() < this->GetSize())
    {
        vtkErrorMacro("The vtkCudaMemoryArray has to little Memory to store memory inside");
        return false;
    }
    // we cannot say the XYZ extent so we just write into the memory area
    if(cudaMemcpyToArray(other->GetArray(),
        0, 0,
        this->GetMemPointer(),
        this->GetSize(),
        cudaMemcpyDeviceToDevice
        ) == cudaSuccess)
        return true;
    else 
        return false;
}

void vtkCudaMemory::PrintSelf (ostream &os, vtkIndent indent)
{
    this->Superclass::PrintSelf(os, indent);
    if (this->GetMemPointer() == NULL)
        os << "Not yet allocated";
}
