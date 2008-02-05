#ifndef CUDAPPMEMORY_H_
#define CUDAPPMEMORY_H_

#include "CudappMemoryBase.h"

class CUDA_SUPPORT_EXPORT CudappMemory : public CudappMemoryBase
{
public:
    virtual ~CudappMemory();
    CudappMemory(const CudappMemory&);
    CudappMemory& operator=(const CudappMemory&); 

    virtual void* AllocateBytes(size_t byte_count) = 0;
    //BTX
    template<typename T> T* Allocate(size_t count) 
        { return (T*)this->AllocateBytes(count * sizeof(T)); }
    //ETX

    void* GetMemPointer() const { return this->MemPointer; }
    //BTX
    template<typename T> T* GetMemPointerAs() const { return (T*)this->GetMemPointer(); }
    //ETX
   
    virtual bool CopyFrom(void* src, size_t byte_count, size_t offset = 0, MemoryLocation src_loc = MemoryOnHost) = 0;
    virtual bool CopyTo(CudappMemoryBase* other) { return other->CopyFrom(this); }

    virtual void PrintSelf(std::ostream &os);

protected:
    CudappMemory();

    void* MemPointer;
    
    bool CopyFrom(CudappMemory* mem);
};

#endif /*CUDAPPMEMORY_H_*/
