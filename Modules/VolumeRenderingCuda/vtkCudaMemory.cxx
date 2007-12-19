#include "vtkCudaMemory.h"

#include "cuda_runtime_api.h"
#include "vtkCudaBase.h"

vtkCudaMemory* vtkCudaMemory::New()
{
  return new vtkCudaMemory();
}


vtkCudaMemory::vtkCudaMemory()
{
  this->MemPointer = NULL;
  this->Size = 0;
}

vtkCudaMemory::~vtkCudaMemory()
{
  // so the virtual function call will not be false.
  // each subclass must call free by its own and set MemPointer to NULL in its Destructor!
  if (this->MemPointer != NULL)
    this->Free();
}

void vtkCudaMemory::Free()
{
  if (this->MemPointer != NULL)
  {
    cudaFree(this->MemPointer);  
    this->MemPointer = NULL;
    this->Size = 0;
  }
}

void* vtkCudaMemory::AllocateBytes(size_t count)
{
  this->Free();
  cudaError_t error = 
    cudaMalloc(&this->MemPointer, count);
  this->Size = count;
  if (error != cudaSuccess)
    vtkCudaBase::PrintError(error);
}

void vtkCudaMemory::MemSet(int value)
{
  cudaMemset(this->MemPointer, value, this->Size);
}
