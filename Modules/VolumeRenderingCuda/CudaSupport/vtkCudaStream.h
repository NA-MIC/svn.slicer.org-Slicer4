#ifndef VTKCUDASTREAM_H_
#define VTKCUDASTREAM_H_

#include "vtkCudaBase.h"

class vtkCudaEvent;

class vtkCudaStream
{
  public:
  static vtkCudaStream* New();
  //BTX
  vtkCudaBase::State e();
  //ETX
  void Synchronize();
  
  cudaStream_t GetStream() const { return this->Stream; }
  
  vtkCudaEvent* GetStreamEvent();
  
protected:
  vtkCudaStream();
  virtual ~vtkCudaStream();

  cudaStream_t Stream;
};

#endif /*VTKCUDASTREAM_H_*/
