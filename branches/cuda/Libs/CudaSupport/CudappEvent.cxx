#include "CudappEvent.h"
#include "CudappStream.h"
#include "CudappBase.h"

#include "cuda_runtime_api.h"

CudappEvent::CudappEvent()
{
    cudaEventCreate(&this->Event);
}

CudappEvent::~CudappEvent()
{
    cudaEventDestroy(this->Event);
}

void CudappEvent::Record()
{
    cudaEventRecord(this->Event, 0);  
}

void CudappEvent::Record(CudappStream* stream)
{
    if (stream == NULL)
        this->Record();
    else
        cudaEventRecord(this->Event, stream->GetStream());
}

CudappBase::State CudappEvent::Query()
{
    switch(cudaEventQuery(this->Event))
    {
    case cudaSuccess:
        return CudappBase::Success;
    case cudaErrorNotReady:
        return CudappBase::NotReadyError;
    case cudaErrorInvalidValue:
    default:
        return CudappBase::InvalidValueError;    
    }
}

void CudappEvent::Synchronize()
{
    if (cudaEventSynchronize(this->Event) == cudaErrorInvalidValue)
        CudappBase::PrintError(cudaErrorInvalidValue);
}

/**
* @returns the time between the finish of two events.
* @param otherEvent the event that finished later than this event (the end event if this is the start-event.
*/
float CudappEvent::ElapsedTime(CudappEvent* otherEvent)
{
    float elapsedTime = 0.0;
    cudaEventElapsedTime(&elapsedTime, this->Event, otherEvent->GetEvent());
    return elapsedTime;
}

void CudappEvent::PrintSelf(std::ostream&  os)
{
}
