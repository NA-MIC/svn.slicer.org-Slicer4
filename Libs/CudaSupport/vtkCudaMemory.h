#ifndef VTKCUDAMEMORY_H_
#define VTKCUDAMEMORY_H_

#include "vtkCudaMemoryBase.h"

class VTK_CUDASUPPORT_EXPORT vtkCudaMemory : public vtkCudaMemoryBase
{
    vtkTypeRevisionMacro(vtkCudaMemory, vtkCudaMemoryBase);
public:

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
    virtual bool CopyTo(vtkCudaMemoryBase* other) { return other->CopyFrom(this); }

    virtual void PrintSelf(ostream &os, vtkIndent indent);

protected:
    vtkCudaMemory();
    virtual ~vtkCudaMemory();
    vtkCudaMemory(const vtkCudaMemory&);
    vtkCudaMemory& operator=(const vtkCudaMemory&); 

    void* MemPointer;
    
    bool CopyFrom(vtkCudaMemory* mem);
};

#endif /*VTKCUDAMEMORY_H_*/
