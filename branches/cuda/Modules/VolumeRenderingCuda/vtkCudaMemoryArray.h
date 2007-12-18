#ifndef VTKCUDAMEMORYARRAY_H_
#define VTKCUDAMEMORYARRAY_H_

#include "channel_descriptor.h"

template<typename T>
class vtkCudaMemoryArray
{
public:
  static vtkCudaMemoryArray* New();
  
  void Allocate(size_t width, size_t height);
  void Free();
  
  void DeepCopy(vtkCudaMemoryArray<T>* source); 
  
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
  size_t Width; //!< The Width of the Array
  size_t Height; //!< The Height of the Array
};

template<typename T>
vtkCudaMemoryArray<T>::vtkCudaMemoryArray()
{
  this->Array = NULL;
  this->Width = this->Height = 0;
  this->Descriptor = cudaCreateChannelDesc<T>();
}

template<typename T>
vtkCudaMemoryArray<T>::~vtkCudaMemoryArray()
{
  this->Free();
}

template<typename T>
vtkCudaMemoryArray<T>* vtkCudaMemoryArray<T>::New()
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
void vtkCudaMemoryArray<T>::Allocate(size_t width, size_t height)
{
  this->Free();
  
  cudaMallocArray(&this->Array, this->Descriptor, width, height);
  this->Width = width;
  this->Height = height;
}

/**
 * @brief frees all the resources needed for the Array
 */
template<typename T>
void vtkCudaMemoryArray<T>::Free()
{
  if (this->Array != NULL) {
    cudaFreeArray(this->Array);  
    this->Array = NULL;
  }
}

template<typename T>
void vtkCudaMemoryArray<T>::DeepCopy(vtkCudaMemoryArray<T>* source)
{
  this->Allocate(source->GetWidth(), source->Getheight());
    cudaMemcpyArrayToArray(this->Array, 0, 0, source->Array, 0, 0, sizeOf(source->Array));
}


#endif /*VTKCUDAMEMORYARRAY_H_*/
