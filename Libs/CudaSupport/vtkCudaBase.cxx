#include "vtkCudaBase.h"

#include "cuda_runtime_api.h"
#include "vtkSetGet.h"

vtkCxxRevisionMacro(vtkCudaBase, "$Revision: 1.0 $");


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
  printf(vtkCudaBase::GetErrorString(error));
}
  

vtkCudaBase* vtkCudaBase::New() 
{
  return NULL;
}

vtkCudaBase::~vtkCudaBase()
{
}

vtkCudaBase::vtkCudaBase()
{
}
