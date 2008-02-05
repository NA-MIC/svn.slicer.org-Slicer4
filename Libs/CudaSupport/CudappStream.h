#ifndef VTKCUDASTREAM_H_
#define VTKCUDASTREAM_H_

#include "CudappBase.h"

class CudappEvent;

class VTK_CUDASUPPORT_EXPORT CudappStream : public vtkObject
{
    vtkTypeRevisionMacro(CudappStream, vtkObject);
public:
    static CudappStream* New();
    //BTX
    CudappBase::State e();
    //ETX
    void Synchronize();

    cudaStream_t GetStream() const { return this->Stream; }

    CudappEvent* GetStreamEvent();

protected:
    CudappStream();
    virtual ~CudappStream();

    cudaStream_t Stream;
};

#endif /*VTKCUDASTREAM_H_*/
