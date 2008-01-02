#include "vtkCudaStream.h"
#include "cuda_runtime_api.h"

#include "vtkCudaEvent.h"

#include "vtkObjectFactory.h"

vtkCxxRevisionMacro(vtkCudaStream, "$Revision: 1.0 $");
vtkStandardNewMacro(vtkCudaStream);

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
vtkCudaEvent* vtkCudaStream::GetStreamEvent()
{
 vtkCudaEvent* event = vtkCudaEvent::New();
 event->Record(this);
 return event;  
}
