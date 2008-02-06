#ifndef CUDAPPDEVICEMEMORY_H_
#define CUDAPPDEVICEMEMORY_H_

#include "CudappMemory.h"
namespace Cudapp
{
    class CUDA_SUPPORT_EXPORT DeviceMemory : public Memory
    {
    public:
        DeviceMemory();
        virtual ~DeviceMemory();
        DeviceMemory(const DeviceMemory&);
        DeviceMemory& operator=(const DeviceMemory&);

        virtual void* AllocateBytes(size_t byte_count);

        virtual void Free();
        virtual void MemSet(int value);

        virtual bool CopyTo(void* dst, size_t byte_count, size_t offset = 0, MemoryLocation dst_loc = MemoryOnHost);
        virtual bool CopyFrom(void* src, size_t byte_count, size_t offset = 0, MemoryLocation src_loc = MemoryOnHost);
        virtual bool CopyTo(MemoryBase* other) { return other->CopyFrom(this->GetMemPointer(), this->GetSize(), 0, MemoryOnDevice); }

        virtual void PrintSelf (std::ostream &os) const;

    protected:

        //virtual bool CopyFrom(MemoryPitch* mem);
        //virtual bool CopyFrom(MemoryArray* mem);

    };
}

#endif /*CUDAPPDEVICEMEMORY_H_*/
