#ifndef CUDAPPEVENT_H_
#define CUDAPPEVENT_H_

#include "CudappBase.h"

class CudappStream;

class CUDA_SUPPORT_EXPORT CudappEvent
{
public:
    CudappEvent();
    virtual ~CudappEvent();

    cudaEvent_t Event;

    //BTX
    void Record();
    void Record(CudappStream* stream);
    CudappBase::State Query();
    //ETX
    void Synchronize();
    float ElapsedTime(CudappEvent* otherEvent);

    /** @returns the Event */
    cudaEvent_t GetEvent() { return this->Event; }

    void PrintSelf(std::ostream&  os);
};

#endif /*CUDAPPEVENT_H_*/
