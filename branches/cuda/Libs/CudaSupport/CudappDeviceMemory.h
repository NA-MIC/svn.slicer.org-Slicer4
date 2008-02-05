#ifndef CUDAPPDEVICEMEMORY_H_
#define CUDAPPDEVICEMEMORY_H_

#include "CudappMemory.h"

class CUDA_SUPPORT_EXPORT CudappDeviceMemory : public CudappMemory
{
public:
    CudappDeviceMemory();
    virtual ~CudappDeviceMemory();
    CudappDeviceMemory(const CudappDeviceMemory&);
    CudappDeviceMemory& operator=(const CudappDeviceMemory&);

    virtual void* AllocateBytes(size_t byte_count);

    virtual void Free();
    virtual void MemSet(int value);
    
    virtual bool CopyTo(void* dst, size_t byte_count, size_t offset = 0, MemoryLocation dst_loc = MemoryOnHost);
    virtual bool CopyFrom(void* src, size_t byte_count, size_t offset = 0, MemoryLocation src_loc = MemoryOnHost);
    virtual bool CopyTo(CudappMemoryBase* other) { return other->CopyFrom(this->GetMemPointer(), this->GetSize(), 0, MemoryOnDevice); }

    virtual void PrintSelf (ostream &os);

protected:

  //virtual bool CopyFrom(CudappMemoryPitch* mem);
  //virtual bool CopyFrom(CudappMemoryArray* mem);

};

#endif /*CUDAPPDEVICEMEMORY_H_*/
