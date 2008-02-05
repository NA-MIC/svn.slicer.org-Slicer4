#ifndef CUDAPPSTREAM_H_
#define CUDAPPSTREAM_H_

#include "CudappBase.h"

class CudappEvent;

class CUDA_SUPPORT_EXPORT CudappStream
{
public:
    CudappStream();
    virtual ~CudappStream();

    //BTX
    CudappBase::State e();
    //ETX
    void Synchronize();

    cudaStream_t GetStream() const { return this->Stream; }
    CudappEvent* GetStreamEvent();

protected:
    cudaStream_t Stream;
};

#endif /*CUDAPPSTREAM_H_*/
