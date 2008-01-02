#include "vtkCudaHostMemory.h"

#include "vtkCudaBase.h"
#include "vtkObjectFactory.h"

#include <string.h>
#include "cuda_runtime_api.h"


vtkCxxRevisionMacro(vtkCudaHostMemory, "$Revision 1.0 $");
vtkStandardNewMacro(vtkCudaHostMemory);

vtkCudaHostMemory::vtkCudaHostMemory()
{
  this->Type = vtkCudaMemoryBase::HostMemory;
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

void vtkCudaHostMemory::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
