#ifndef VTKCUDAEVENT_H_
#define VTKCUDAEVENT_H_

#include "CudappBase.h"

class CudappStream;

class VTK_CUDASUPPORT_EXPORT CudappEvent : public vtkObject
{
    vtkTypeRevisionMacro(CudappEvent, vtkObject);
public:
    static CudappEvent* New();

    //BTX
    void Record();
    void Record(CudappStream* stream);
    CudappBase::State Query();
    //ETX
    void Synchronize();
    float ElapsedTime(CudappEvent* otherEvent);

    /** @returns the Event */
    cudaEvent_t GetEvent() { return this->Event; }

    void PrintSelf(ostream& os, vtkIndent indent);

protected:
    CudappEvent();
    virtual ~CudappEvent();

    cudaEvent_t Event;
};

#endif /*VTKCUDAEVENT_H_*/
