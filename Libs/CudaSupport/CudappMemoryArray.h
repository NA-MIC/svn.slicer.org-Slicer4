#ifndef VTKCUDAMEMORYARRAY_H_
#define VTKCUDAMEMORYARRAY_H_

#include "CudappMemoryBase.h"
#include "channel_descriptor.h"

class VTK_CUDASUPPORT_EXPORT CudappMemoryArray : public CudappMemoryBase
{
    vtkTypeRevisionMacro(CudappMemoryArray, CudappMemoryBase);
public:
    static CudappMemoryArray* New();

    //BTX
    template<typename T>
      void SetFormat() { this->Descriptor = cudaCreateChannelDesc<T>(); }
    //ETX
    void SetChannelDescriptor(const cudaChannelFormatDesc& desc) { this->Descriptor = desc; }

    void Allocate(size_t width, size_t height);
    virtual void Free();
    virtual void MemSet(int value) {}

    void DeepCopy(CudappMemoryArray* source); 

    const cudaChannelFormatDesc& GetDescriptor() const { return this->Descriptor; } 
    cudaArray* GetArray() const { return this->Array; }

    size_t GetWidth() const { return this->Width; }
    size_t GetHeight() const { return this->Height; }

  //HACK    
    virtual bool CopyTo(void* dst, size_t byte_count, size_t offset = 0, MemoryLocation dst_loc = MemoryOnHost) { return false; }
    virtual bool CopyFrom(void* src, size_t byte_count, size_t offset = 0, MemoryLocation src_loc = MemoryOnHost) { return false; }

    virtual void PrintSelf(ostream &os, vtkIndent indent);

protected:
    CudappMemoryArray();
    virtual ~CudappMemoryArray();
    CudappMemoryArray(const CudappMemoryArray&);
    CudappMemoryArray& operator=(const CudappMemoryArray&);

    cudaChannelFormatDesc Descriptor; //!< The Descriptor used to allocate memory
    cudaArray* Array; //!< The Array with the memory that was allocated.
    size_t Width;  //!< The Width of the Array
    size_t Height; //!< The Height of the Array
};



#endif /*VTKCUDAMEMORYARRAY_H_*/
