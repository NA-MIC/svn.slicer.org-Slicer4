#ifndef VTKCUDAEVENT_H_
#define VTKCUDAEVENT_H_

#include "vtkCudaBase.h"

class vtkCudaStream;

class VTK_CUDASUPPORTMODULE_EXPORT vtkCudaEvent
{
public:
  static vtkCudaEvent* New();
  
  //BTX
  void Record(vtkCudaStream* stream = NULL);
  vtkCudaBase::State Query();
  //ETX
  void Synchronize();
  float ElapsedTime(vtkCudaEvent* otherEvent);
  
  /** @returns the Event */
  cudaEvent_t GetEvent() { return this->Event; }
  
  /// public for now.  
  virtual ~vtkCudaEvent();
protected:
  vtkCudaEvent();
  
  cudaEvent_t Event;
};

#endif /*VTKCUDAEVENT_H_*/
