#include "vtkCudaDeviceMemory.h"

#include "cuda_runtime_api.h"
#include "vtkCudaBase.h"

#include "vtkCudaLocalMemory.h"
#include "vtkCudaHostMemory.h"
#include "vtkCudaMemoryArray.h"

#include "vtkObjectFactory.h"
#include "vtkImageData.h"

vtkCxxRevisionMacro(vtkCudaDeviceMemory, "$Revision: 1.0 $");
vtkStandardNewMacro(vtkCudaDeviceMemory);

vtkCudaDeviceMemory::vtkCudaDeviceMemory()
{
    this->MemPointer = NULL;
    this->Size = 0;
}

vtkCudaDeviceMemory::~vtkCudaDeviceMemory()
{
    // so the virtual function call will not be false.
    // each subclass must call free by its own and set MemPointer to NULL in its Destructor!
    if (this->MemPointer != NULL)
        this->Free();
}

void vtkCudaDeviceMemory::Free()
{
    if (this->MemPointer != NULL)
    {
        cudaFree(this->MemPointer);  
        this->MemPointer = NULL;
        this->Size = 0;
    }
}

void* vtkCudaDeviceMemory::AllocateBytes(size_t byte_count)
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

void vtkCudaDeviceMemory::MemSet(int value)
{
    cudaMemset(this->MemPointer, value, this->Size);
}

bool vtkCudaDeviceMemory::CopyFrom(vtkImageData* data)
{
    if (!this->AllocateBytes(data->GetActualMemorySize()*1024))
        return false;

    if(cudaMemcpy(this->GetMemPointer(),
        data->GetScalarPointer(),
        this->GetSize(),
        cudaMemcpyHostToDevice
        ) == cudaSuccess)
        return true;
    else 
        return false;
}

bool vtkCudaDeviceMemory::CopyTo(vtkImageData* data)
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

bool vtkCudaDeviceMemory::CopyTo(vtkCudaDeviceMemory* other)
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

bool vtkCudaDeviceMemory::CopyTo(vtkCudaLocalMemory* other)
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

bool vtkCudaDeviceMemory::CopyTo(vtkCudaMemoryArray* other)
{
    if (other->GetSize() < this->GetSize())
    {
        vtkErrorMacro("The vtkCudaDeviceMemoryArray has to little Memory to store memory inside");
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

void vtkCudaDeviceMemory::PrintSelf (ostream &os, vtkIndent indent)
{
    this->Superclass::PrintSelf(os, indent);
    if (this->GetMemPointer() == NULL)
        os << "Not yet allocated";
}
