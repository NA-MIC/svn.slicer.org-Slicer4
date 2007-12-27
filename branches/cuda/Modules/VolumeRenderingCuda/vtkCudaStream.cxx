#include "vtkCudaStream.h"
#include "cuda_runtime_api.h"

#include "vtkCudaEvent.h"

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

/**
 * @brief Creates and returns a new vtkCudaEvent that triggers when the Stream is finished.
 * @returns a new vtkCudaEvent triggering on this Stream.
 */
vtkCudaEvent* vtkCudaStream::GetStreamEvent() const
{
vtkCudaEvent* event = vtkCudaEvent::New();
event->Record(this);
return event;  
}
