#ifndef CUDAPPMEMORY_H_
#define CUDAPPMEMORY_H_

#include "CudappMemoryBase.h"
namespace Cudapp
{
    class CUDA_SUPPORT_EXPORT Memory : public MemoryBase
    {
    public:
        virtual ~Memory();
        Memory(const Memory&);
        Memory& operator=(const Memory&);

        virtual void* AllocateBytes(size_t byte_count) = 0;
        //BTX
        template<typename T> T* Allocate(size_t count) 
        { return (T*)this->AllocateBytes(count * sizeof(T)); }
        //ETX
        void* GetMemPointer() { return this->MemPointer; }
        const void* GetMemPointer() const { return this->MemPointer; }
        //BTX
        template<typename T> T* GetMemPointerAs() { return (T*)this->GetMemPointer(); }
        template<typename T> const T* GetMemPointerAs() const { return (T*)this->GetMemPointer(); }
        //ETX

        virtual bool CopyFrom(const void* src, size_t byte_count, size_t offset = 0, MemoryLocation src_loc = MemoryOnHost) = 0;
        virtual bool CopyTo(MemoryBase* other) const { return other->CopyFromInternal(this); }

        virtual void PrintSelf(std::ostream &os) const;

    protected:
        Memory();
        virtual bool CopyFromInternal(const Memory* mem);

    protected:
        void* MemPointer;
    };
}
#endif /*CUDAPPMEMORY_H_*/
