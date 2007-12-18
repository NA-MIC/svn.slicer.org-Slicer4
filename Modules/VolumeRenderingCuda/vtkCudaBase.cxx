#include "vtkCudaBase.h"

#include "cuda_runtime_api.h"

cudaError_t vtkCudaBase::GetLastError()
{
  return cudaGetLastError();
}

const char* vtkCudaBase::GetLastErrorString()
{
  vtkCudaBase::GetErrorString(vtkCudaBase::GetLastError());
}

const char* vtkCudaBase::GetErrorString(cudaError_t error)
{
  return cudaGetErrorString(error);
}

void vtkCudaBase::PrintError(cudaError_t error)
{
  vtkDebug(vtkCudaBase::GetErrorString(error));
}
  

vtkCudaBase::~vtkCudaBase()
{
}

vtkCudaBase::vtkCudaBase()
{
}
