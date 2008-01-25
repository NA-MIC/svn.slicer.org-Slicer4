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

    virtual void Free() = 0;
    virtual void MemSet(int value) = 0;

    void* GetMemPointer() const { return this->MemPointer; }
    //BTX
    template<typename T> T* GetMemPointerAs() const { return (T*)this->GetMemPointer(); }
    //ETX

    virtual bool CopyFrom(vtkImageData* data) { return false ; }
    virtual bool CopyTo(vtkImageData* data) { return false; }
    virtual bool CopyTo(vtkCudaMemory* other) { return false; }
    virtual bool CopyTo(vtkCudaLocalMemory* other) { return false; }
    virtual bool CopyTo(vtkCudaMemoryArray* other) { return false; }

    virtual void PrintSelf (ostream &os, vtkIndent indent);

protected:
    vtkCudaMemory();
    virtual ~vtkCudaMemory();
    vtkCudaMemory(const vtkCudaMemory&);
    vtkCudaMemory& operator=(const vtkCudaMemory&);

    void* MemPointer;
};

#endif /*VTKCUDAMEMORY_H_*/
