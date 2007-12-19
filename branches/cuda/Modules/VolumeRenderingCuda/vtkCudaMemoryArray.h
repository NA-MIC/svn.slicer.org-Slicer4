#ifndef VTKCUDAMEMORYARRAY_H_
#define VTKCUDAMEMORYARRAY_H_

#include "vtkCudaMemory.h"
#include "channel_descriptor.h"

class vtkCudaMemoryArray : public vtkCudaMemory
{
public:
  static vtkCudaMemoryArray* New();
  
  //BTX
  template<typename T>
    void SetFormat() { this->Descriptor = cudaCreateChannelDesc<T>(); }
  //ETX
  void SetChannelDescriptor(const cudaChannelFormatDesc& desc) { this->Descriptor = desc; }
  
  void Allocate(size_t width, size_t height);
  virtual void Free();
  
  void DeepCopy(vtkCudaMemoryArray* source); 
  
  const cudaChannelFormatDesc& GetDescriptor() const { return this->Descriptor; } 
  cudaArray* GetArray() const { return this->Array; }

  size_t GetWidth() const { return this->Width; }
  size_t GetHeight() const { return this->Height; }

protected:
  vtkCudaMemoryArray();
  virtual ~vtkCudaMemoryArray();
  vtkCudaMemoryArray(const vtkCudaMemoryArray&);
  vtkCudaMemoryArray operator=(const vtkCudaMemoryArray&);
  
  cudaChannelFormatDesc Descriptor; //!< The Descriptor used to allocate memory
  cudaArray* Array; //!< The Array with the memory that was allocated.
  size_t Width;  //!< The Width of the Array
  size_t Height; //!< The Height of the Array
};



#endif /*VTKCUDAMEMORYARRAY_H_*/
