#ifndef CUDAPPLOCALMEMORY_H_
#define CUDAPPLOCALMEMORY_H_

#include "CudappMemory.h"

namespace Cudapp
{
    class CUDA_SUPPORT_EXPORT LocalMemory : public Memory
    {
    public:
        LocalMemory();
        virtual ~LocalMemory();
        LocalMemory(const LocalMemory&);
        LocalMemory& operator=(const LocalMemory&);

        virtual void* AllocateBytes(size_t count);
        virtual void Free();
        virtual void MemSet(int value); 

        virtual bool CopyTo(void* dst, size_t byte_count, size_t offset = 0, MemoryLocation dst_loc = MemoryOnHost);
        virtual bool CopyFrom(void* src, size_t byte_count, size_t offset = 0, MemoryLocation src_loc = MemoryOnHost);
        virtual bool CopyTo(MemoryBase* other) { return other->CopyFrom(this->GetMemPointer(), this->GetSize(), 0, MemoryOnHost); }

        void PrintSelf(std::ostream&  os);

    protected:

        //virtual bool CopyFrom(MemoryPitch* mem);
        //virtual bool CopyFrom(MemoryArray* mem);

    };
}
#endif /* CUDAPPLOCALMEMORY_H_ */
