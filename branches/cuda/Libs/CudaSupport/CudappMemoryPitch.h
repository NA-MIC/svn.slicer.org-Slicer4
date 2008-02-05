#ifndef VTKCUDAMEMORYPITCH_H_
#define VTKCUDAMEMORYPITCH_H_

#include "CudappMemoryBase.h"

class VTK_CUDASUPPORT_EXPORT CudappMemoryPitch : public CudappMemoryBase
{
    vtkTypeRevisionMacro(CudappMemoryPitch, CudappMemoryBase);
public:
    static CudappMemoryPitch* New();

    void* AllocatePitchBytes(size_t width, size_t height, size_t typeSize);
  virtual void Free();

    virtual void MemSet(int value);

    //BTX
    template<typename T> T* AllocatePitch(size_t width, size_t height)
      { return (T*)this->AllocatePitchBytes(width, height, sizeof(T)); }
    //ETX

  void* GetMemPointer() const { return this->MemPointer; }
    size_t GetPitch() const { return this->Pitch; }
    size_t GetWidth() const { return this->Width; } 
    size_t GetHeight() const { return this->Height; }
    
    //HACK    
    virtual bool CopyTo(void* dst, size_t byte_count, size_t offset = 0, MemoryLocation dst_loc = MemoryOnHost) { return false; }
    virtual bool CopyFrom(void* src, size_t byte_count, size_t offset = 0, MemoryLocation src_loc = MemoryOnHost) { return false; }
    

    virtual void PrintSelf (ostream &os, vtkIndent indent);
protected:
    CudappMemoryPitch();
    virtual ~CudappMemoryPitch();
    CudappMemoryPitch(const CudappMemoryPitch&);
    CudappMemoryPitch& operator=(const CudappMemoryPitch&);

    size_t Pitch;
    size_t Width;
    size_t Height;
    
    void* MemPointer;
};

#endif /*VTKCUDAMEMORYPITCH_H_*/
