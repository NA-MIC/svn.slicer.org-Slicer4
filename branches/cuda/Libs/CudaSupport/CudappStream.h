#ifndef CUDAPPSTREAM_H_
#define CUDAPPSTREAM_H_

#include "CudappBase.h"
namespace Cudapp
{
    class Event;
    class CUDA_SUPPORT_EXPORT Stream
    {
    public:
        Stream();
        virtual ~Stream();

        //BTX
        Base::State e();
        //ETX
        void Synchronize();

        cudaStream_t GetStream() const { return this->CudaStream; }
        Event* GetStreamEvent();

    protected:
        cudaStream_t CudaStream;
    };
}
#endif /*CUDAPPSTREAM_H_*/
