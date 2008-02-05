#ifndef CUDAPPLOCALMEMORY_H_
#define CUDAPPLOCALMEMORY_H_

#include "CudappMemory.h"

class CUDA_SUPPORT_EXPORT CudappLocalMemory : public CudappMemory
{
public:
    CudappLocalMemory();
    virtual ~CudappLocalMemory();
    CudappLocalMemory(const CudappLocalMemory&);
    CudappLocalMemory& operator=(const CudappLocalMemory&);

    virtual void* AllocateBytes(size_t count);
    virtual void Free();
    virtual void MemSet(int value); 

    virtual bool CopyTo(void* dst, size_t byte_count, size_t offset = 0, MemoryLocation dst_loc = MemoryOnHost);
    virtual bool CopyFrom(void* src, size_t byte_count, size_t offset = 0, MemoryLocation src_loc = MemoryOnHost);
    virtual bool CopyTo(CudappMemoryBase* other) { return other->CopyFrom(this->GetMemPointer(), this->GetSize(), 0, MemoryOnHost); }
    
    void PrintSelf(std::ostream&  os);

protected:
    
  //virtual bool CopyFrom(CudappMemoryPitch* mem);
  //virtual bool CopyFrom(CudappMemoryArray* mem);
    
};

#endif /* CUDAPPLOCALMEMORY_H_ */
