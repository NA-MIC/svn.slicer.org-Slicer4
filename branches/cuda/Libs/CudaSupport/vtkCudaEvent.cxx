#include "vtkCudaEvent.h"
#include "vtkCudaStream.h"
#include "vtkCudaBase.h"

#include "cuda_runtime_api.h"

vtkCudaEvent* vtkCudaEvent::New()
{
  return new vtkCudaEvent();  
}

vtkCudaEvent::vtkCudaEvent()
{
  cudaEventCreate(&this->Event);
}

vtkCudaEvent::~vtkCudaEvent()
{
  cudaEventDestroy(this->Event);
}

void vtkCudaEvent::Record(vtkCudaStream* stream)
{
  if (stream == NULL)
   cudaEventRecord(this->Event, 0);
  else
   cudaEventRecord(this->Event, stream->GetStream());
}

vtkCudaBase::State vtkCudaEvent::Query()
{
  switch(cudaEventQuery(this->Event))
  {
  case cudaSuccess:
    return vtkCudaBase::Success;
  case cudaErrorNotReady:
    return vtkCudaBase::NotReadyError;
  case cudaErrorInvalidValue:
  default:
    return vtkCudaBase::InvalidValueError;    
  }
}

void vtkCudaEvent::Synchronize()
{
  if (cudaEventSynchronize(this->Event) == cudaErrorInvalidValue)
    vtkCudaBase::PrintError(cudaErrorInvalidValue);
}

/**
 * @returns the time between the finish of two events.
 * @param otherEvent the event that finished later than this event (the end event if this is the start-event.
 */
float vtkCudaEvent::ElapsedTime(vtkCudaEvent* otherEvent)
{
  float elapsedTime = 0.0;
  cudaEventElapsedTime(&elapsedTime, this->Event, otherEvent->GetEvent());
  return elapsedTime;
}
