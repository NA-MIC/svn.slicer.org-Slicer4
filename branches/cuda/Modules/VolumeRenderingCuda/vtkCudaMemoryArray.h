#ifndef VTKCUDAMEMORYARRAY_H_
#define VTKCUDAMEMORYARRAY_H_

#include "channel_descriptor.h"

template<typename T>
class vtkCudaMemoryArray
{
  public:
  vtkCudaMemoryArray* New();
  
  void Allocate(size_t width, size_t height);
  void Free();
  
  const cudaChannelFormatDescriptor& GetDescriptor() const { return this->Descriptor; } 
  cudaArray* GetArray() const { return this->Array; }
  
protected:
  vtkCudaMemoryArray();
  virtual ~vtkCudaMemoryArray();
  vtkCudaMemoryArray(const vtkCudaMemoryArray&);
  vtkCudaMemoryArray operator=(const vtkCudaMemoryArray&);
  
  cudaChannelFormatDescriptor Descriptor; //!< The Descriptor used to allocate memory
  cudaArray* Array; //!< The Array with the memory that was allocated.
  size_t Width; //!< The Width of the Array
  size_t Height; //!< The Height of the Array
};

template<typename T>
vtkCudaMemoryArray::vtkCudaMemoryArray()
{
  this->Array = NULL;
  this->Width = this->Height = 0;
  this->Descriptor = cudaCreateFormatDescriptor<T>();
}

template<typename T>
vtkCudaMemoryArray::~vtkCudaMemoryArray()
{
  this->Free();
}

template<typename T>
vtkCudaMemoryArray* vtkCudaMemoryArray::New()
{
  return new vtkCudaMemoryArray();
}

/**
 * Allocates A new array of size width*height with the specified type T
 * @param width the width of the array to allocate
 * @param height the height of the array to allocate.
 * 
 * @note if there was already allocated data in this instance the data will be erased.
 */
template<typename T>
void vtkCudaMemoryArray::Allocate(size_t width, size_t height)
{
  this->Free();
  
  cudaMallocArray(&this->Array, this->Descriptor, width, height);
  this->Width = width;
  this->Height = height;
}

/**
template<typename T>
void vtkCudaMemoryArray::Free()
{
  if (this->Array != NULL) {
    cudaFreeArray(this->Array);  
    this->Array = NULL;
  }
}

#endif /*VTKCUDAMEMORYARRAY_H_*/
