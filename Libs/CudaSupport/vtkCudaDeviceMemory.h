#ifndef VTKCUDADEVICEMEMORY_H_
#define VTKCUDADEVICEMEMORY_H_

#include "vtkCudaMemory.h"

class VTK_CUDASUPPORT_EXPORT vtkCudaDeviceMemory : public vtkCudaMemory
{
    vtkTypeRevisionMacro(vtkCudaDeviceMemory, vtkCudaMemory);
public:
    static  vtkCudaDeviceMemory* New();

    virtual void* AllocateBytes(size_t byte_count);

    virtual void Free();
    virtual void MemSet(int value);
    
    virtual bool CopyTo(void* dst, size_t byte_count, size_t offset = 0, MemoryLocation dst_loc = MemoryOnHost);
    virtual bool CopyFrom(void* src, size_t byte_count, size_t offset = 0, MemoryLocation src_loc = MemoryOnHost);
    virtual bool CopyTo(vtkCudaMemoryBase* other) { return this->Superclass::CopyFrom(this); }

    virtual void PrintSelf (ostream &os, vtkIndent indent);

protected:
    vtkCudaDeviceMemory();
    virtual ~vtkCudaDeviceMemory();
    vtkCudaDeviceMemory(const vtkCudaDeviceMemory&);
    vtkCudaDeviceMemory& operator=(const vtkCudaDeviceMemory&);

  //virtual bool CopyFrom(vtkCudaMemoryPitch* mem);
  //virtual bool CopyFrom(vtkCudaMemoryArray* mem);

};

#endif /*VTKCUDADEVICEMEMORY_H_*/
