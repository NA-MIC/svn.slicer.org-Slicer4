#ifndef VTKCUDALOCALMEMORY_H_
#define VTKCUDALOCALMEMORY_H_

#include "vtkCudaMemory.h"

class VTK_CUDASUPPORT_EXPORT vtkCudaLocalMemory : public vtkCudaMemory
{
    vtkTypeRevisionMacro(vtkCudaLocalMemory, vtkCudaMemory);
public:
    static vtkCudaLocalMemory* New();

    virtual void* AllocateBytes(size_t count);
    virtual void Free();
    virtual void MemSet(int value); 

    virtual bool CopyTo(void* dst, size_t byte_count, size_t offset = 0, MemoryLocation dst_loc = MemoryOnHost);
    virtual bool CopyFrom(void* src, size_t byte_count, size_t offset = 0, MemoryLocation src_loc = MemoryOnHost);
    virtual bool CopyTo(vtkCudaMemoryBase* other) { return this->Superclass::CopyFrom(this); }
    
    void PrintSelf(ostream& os, vtkIndent indent);

protected:
    vtkCudaLocalMemory();
    virtual ~vtkCudaLocalMemory();
    vtkCudaLocalMemory(const vtkCudaLocalMemory&);
    vtkCudaLocalMemory& operator=(const vtkCudaLocalMemory&);
    
  //virtual bool CopyFrom(vtkCudaMemoryPitch* mem);
  //virtual bool CopyFrom(vtkCudaMemoryArray* mem);
    
};

#endif /* VTKCUDALOCALMEMORY_H_ */
