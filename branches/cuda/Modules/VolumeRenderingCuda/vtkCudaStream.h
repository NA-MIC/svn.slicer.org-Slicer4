#ifndef VTKCUDASTREAM_H_
#define VTKCUDASTREAM_H_

#include "vtkCudaBase.h"

class vtkCudaStream
{
  public:
  static vtkCudaStream* New();
  //BTX
  vtkCudaBase::State Query();
  //ETX
  void Synchronize();
  
  cudaStream_t GetStream() const { return this->Stream; }
  
protected:
  vtkCudaStream();
  virtual ~vtkCudaStream();

  cudaStream_t Stream;
};

#endif /*VTKCUDASTREAM_H_*/
