#include "vtkCudaMemoryBase.h"

#include "vtkCudaMemory.h"

vtkCudaMemoryBase* vtkCudaMemoryBase::New()
{
  return vtkCudaMemory::New();
}

vtkCudaMemoryBase::~vtkCudaMemoryBase()
{
}

vtkCudaMemoryBase::vtkCudaMemoryBase()
{
  this->Type = vtkCudaMemoryBase::Undefined;
}
