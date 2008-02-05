#include "CudappStream.h"
#include "cuda_runtime_api.h"

#include "CudappEvent.h"

#include "vtkObjectFactory.h"

vtkCxxRevisionMacro(CudappStream, "$Revision: 1.0 $");
vtkStandardNewMacro(CudappStream);

CudappStream::CudappStream()
{
    cudaStreamCreate(&this->Stream);
}

CudappStream::~CudappStream()
{
    cudaStreamDestroy(this->Stream);
}

void CudappStream::Synchronize()
{
    cudaStreamSynchronize(this->Stream);  
}

/**
* @brief Creates and returns a new CudappEvent that triggers when the Stream is finished.
* @returns a new CudappEvent triggering on this Stream.
*/
CudappEvent* CudappStream::GetStreamEvent()
{
    CudappEvent* event = CudappEvent::New();
    event->Record(this);
    return event;  
}
