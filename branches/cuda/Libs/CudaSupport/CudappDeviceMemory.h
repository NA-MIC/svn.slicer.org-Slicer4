#ifndef VTKCUDADEVICEMEMORY_H_
#define VTKCUDADEVICEMEMORY_H_

#include "CudappMemory.h"

class VTK_CUDASUPPORT_EXPORT CudappDeviceMemory : public CudappMemory
{
    vtkTypeRevisionMacro(CudappDeviceMemory, CudappMemory);
public:
    static  CudappDeviceMemory* New();

    virtual void* AllocateBytes(size_t byte_count);

    virtual void Free();
    virtual void MemSet(int value);
    
    virtual bool CopyTo(void* dst, size_t byte_count, size_t offset = 0, MemoryLocation dst_loc = MemoryOnHost);
    virtual bool CopyFrom(void* src, size_t byte_count, size_t offset = 0, MemoryLocation src_loc = MemoryOnHost);
    virtual bool CopyTo(CudappMemoryBase* other) { return this->Superclass::CopyFrom(this); }

    virtual void PrintSelf (ostream &os, vtkIndent indent);

protected:
    CudappDeviceMemory();
    virtual ~CudappDeviceMemory();
    CudappDeviceMemory(const CudappDeviceMemory&);
    CudappDeviceMemory& operator=(const CudappDeviceMemory&);

  //virtual bool CopyFrom(CudappMemoryPitch* mem);
  //virtual bool CopyFrom(CudappMemoryArray* mem);

};

#endif /*VTKCUDADEVICEMEMORY_H_*/
