#include "CudappDeviceMemory.h"

#include "cuda_runtime_api.h"
#include "CudappBase.h"

#include "CudappLocalMemory.h"
#include "CudappHostMemory.h"
#include "CudappMemoryArray.h"

#include "vtkObjectFactory.h"
#include "vtkImageData.h"

vtkCxxRevisionMacro(CudappDeviceMemory, "$Revision: 1.0 $");
vtkStandardNewMacro(CudappDeviceMemory);

CudappDeviceMemory::CudappDeviceMemory()
{
  this->Location = CudappMemoryBase::MemoryOnDevice;
  
  
    this->MemPointer = NULL;
    this->Size = 0;
}

CudappDeviceMemory::~CudappDeviceMemory()
{
    // so the virtual function call will not be false.
    // each subclass must call free by its own and set MemPointer to NULL in its Destructor!
    if (this->MemPointer != NULL)
        this->Free();
}

void CudappDeviceMemory::Free()
{
    if (this->MemPointer != NULL)
    {
        cudaFree(this->MemPointer);  
        this->MemPointer = NULL;
        this->Size = 0;
    }
}

void* CudappDeviceMemory::AllocateBytes(size_t byte_count)
{
    // do nothing in case we already allocated the desired size.
    if (this->GetSize() == byte_count)
        return this->MemPointer;

    this->Free();
    cudaError_t error = 
        cudaMalloc(&this->MemPointer, byte_count);
    this->Size = byte_count;
    if (error != cudaSuccess)
        CudappBase::PrintError(error);

    return (void*) this->MemPointer;
}

void CudappDeviceMemory::MemSet(int value)
{
    cudaMemset(this->MemPointer, value, this->Size);
}

bool CudappDeviceMemory::CopyTo(void* dst, size_t byte_count, size_t offset, MemoryLocation dst_loc)
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

bool CudappDeviceMemory::CopyFrom(void* src, size_t byte_count, size_t offset, MemoryLocation src_loc)
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
bool CudappDeviceMemory::CopyFrom(vtkImageData* data)
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

bool CudappDeviceMemory::CopyTo(vtkImageData* data)
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
bool CudappDeviceMemory::CopyTo(CudappMemoryArray* other)
{
    if (other->GetSize() < this->GetSize())
    {
        vtkErrorMacro("The CudappDeviceMemoryArray has to little Memory to store memory inside");
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

void CudappDeviceMemory::PrintSelf (ostream &os, vtkIndent indent)
{
    this->Superclass::PrintSelf(os, indent);
}
