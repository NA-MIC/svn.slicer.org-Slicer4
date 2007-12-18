#include "vtkCudaStream.h"
#include "cuda_runtime_api.h"

vtkCudaStream* vtkCudaStream::New()
{
  return new vtkCudaStream();  
}

vtkCudaStream::vtkCudaStream()
{
  cudaStreamCreate(&this->Stream);
}

vtkCudaStream::~vtkCudaStream()
{
  cudaStreamDestroy(this->Stream);
}

void vtkCudaStream::Synchronize()
{
  cudaStreamSynchronize(this->Stream);  
}
