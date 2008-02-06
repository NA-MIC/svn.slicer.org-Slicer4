#include "CudappDeviceMemory.h"

#include "cuda_runtime_api.h"
#include "CudappBase.h"

namespace Cudapp
{
    DeviceMemory::DeviceMemory()
    {
        this->Location = MemoryBase::MemoryOnDevice;


        this->MemPointer = NULL;
        this->Size = 0;
    }

    DeviceMemory::~DeviceMemory()
    {
        // so the virtual function call will not be false.
        // each subclass must call free by its own and set MemPointer to NULL in its Destructor!
        if (this->MemPointer != NULL)
            this->Free();
    }

    void DeviceMemory::Free()
    {
        if (this->MemPointer != NULL)
        {
            cudaFree(this->MemPointer);  
            this->MemPointer = NULL;
            this->Size = 0;
        }
    }

    void* DeviceMemory::AllocateBytes(size_t byte_count)
    {
        // do nothing in case we already allocated the desired size.
        if (this->GetSize() == byte_count)
            return this->MemPointer;

        this->Free();
        cudaError_t error = 
            cudaMalloc(&this->MemPointer, byte_count);
        this->Size = byte_count;
        if (error != cudaSuccess)
            Base::PrintError(error);

        return (void*) this->MemPointer;
    }

    void DeviceMemory::MemSet(int value)
    {
        cudaMemset(this->MemPointer, value, this->Size);
    }

    bool DeviceMemory::CopyTo(void* dst, size_t byte_count, size_t offset, MemoryLocation dst_loc) const
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

    bool DeviceMemory::CopyFrom(const void* src, size_t byte_count, size_t offset, MemoryLocation src_loc)
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
    bool DeviceMemory::CopyFrom(vtkImageData* data)
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

    bool DeviceMemory::CopyTo(vtkImageData* data)
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
    bool DeviceMemory::CopyTo(MemoryArray* other)
    {
    if (other->GetSize() < this->GetSize())
    {
    vtkErrorMacro("The DeviceMemoryArray has to little Memory to store memory inside");
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

    void DeviceMemory::PrintSelf (std::ostream &os) const
    {
        this->Memory::PrintSelf(os);
    }
}
