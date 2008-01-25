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
  this->Location = vtkCudaMemoryBase::MemoryOnDevice;
  
  
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

bool vtkCudaDeviceMemory::CopyTo(void* dst, size_t byte_count, size_t offset, MemoryLocation dst_loc)
{
  if(cudaMemcpy(dst, 
        this->GetMemPointer(), //HACK + offset,
        byte_count,
        (dst_loc == MemoryOnHost) ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice
        ) == cudaSuccess)
        return true;
    else 
        return false;
}

bool vtkCudaDeviceMemory::CopyFrom(void* src, size_t byte_count, size_t offset, MemoryLocation src_loc)
{
    if(cudaMemcpy(this->GetMemPointer(), //HACK + offset, 
        src,
        byte_count,
        (src_loc == MemoryOnHost) ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice
        ) == cudaSuccess)
        return true;
    else 
        return false;
}


/* vtkImageData
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
*/
/* ARRAY
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
*/

void vtkCudaDeviceMemory::PrintSelf (ostream &os, vtkIndent indent)
{
    this->Superclass::PrintSelf(os, indent);
}
