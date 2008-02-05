#ifndef VTKCUDALOCALMEMORY_H_
#define VTKCUDALOCALMEMORY_H_

#include "CudappMemory.h"

class VTK_CUDASUPPORT_EXPORT CudappLocalMemory : public CudappMemory
{
    vtkTypeRevisionMacro(CudappLocalMemory, CudappMemory);
public:
    static CudappLocalMemory* New();

    virtual void* AllocateBytes(size_t count);
    virtual void Free();
    virtual void MemSet(int value); 

    virtual bool CopyTo(void* dst, size_t byte_count, size_t offset = 0, MemoryLocation dst_loc = MemoryOnHost);
    virtual bool CopyFrom(void* src, size_t byte_count, size_t offset = 0, MemoryLocation src_loc = MemoryOnHost);
    virtual bool CopyTo(CudappMemoryBase* other) { return other->CopyFrom(this->GetMemPointer(), this->GetSize(), 0, MemoryOnHost); }
    
    void PrintSelf(ostream& os, vtkIndent indent);

protected:
    CudappLocalMemory();
    virtual ~CudappLocalMemory();
    CudappLocalMemory(const CudappLocalMemory&);
    CudappLocalMemory& operator=(const CudappLocalMemory&);
    
  //virtual bool CopyFrom(CudappMemoryPitch* mem);
  //virtual bool CopyFrom(CudappMemoryArray* mem);
    
};

#endif /* VTKCUDALOCALMEMORY_H_ */
