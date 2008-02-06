#ifndef CUDAPPEVENT_H_
#define CUDAPPEVENT_H_

#include "CudappBase.h"
namespace Cudapp
{
    class Stream;
    class CUDA_SUPPORT_EXPORT Event
    {
    public:
        Event();
        virtual ~Event();

        //BTX
        void Record();
        void Record(Stream* stream);
        Base::State Query();
        //ETX
        void Synchronize();
        float ElapsedTime(Event* otherEvent);

        /** @returns the Event */
        cudaEvent_t GetEvent() { return this->CudaEvent; }

        void PrintSelf(std::ostream&  os);

    private:
        cudaEvent_t CudaEvent;
    };
}
#endif /*CUDAPPEVENT_H_*/
