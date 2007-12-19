#include "vtkCudaHostMemory.h"

#include "vtkCudaBase.h"

#include <string.h>
#include "cuda_runtime_api.h"


vtkCudaHostMemory* vtkCudaHostMemory::New()
{
  return new vtkCudaHostMemory();  
}

vtkCudaHostMemory::vtkCudaHostMemory()
{
}

vtkCudaHostMemory::~vtkCudaHostMemory()
{
  this->Free();
}


void* vtkCudaHostMemory::AllocateBytes(size_t count)
{
  this->Free();
  cudaError_t error = 
    cudaMallocHost(&this->MemPointer, count);
  this->Size = count;
  if (error != cudaSuccess)
    vtkCudaBase::PrintError(error);
}

void vtkCudaHostMemory::Free()
{
  if (this->MemPointer != NULL)
  {
    cudaFreeHost(this->MemPointer);
    this->MemPointer = NULL;  
    this->Size = 0;
  }
}

/**
 * @brief host implementation of the MemorySetter Value
 */
void vtkCudaHostMemory::MemSet(int value)
{
  memset(this->MemPointer, value, Size);
}
