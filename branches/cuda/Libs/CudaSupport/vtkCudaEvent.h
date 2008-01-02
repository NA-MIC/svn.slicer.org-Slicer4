#ifndef VTKCUDAEVENT_H_
#define VTKCUDAEVENT_H_

#include "vtkCudaBase.h"

class vtkCudaStream;

class VTK_CUDASUPPORT_EXPORT vtkCudaEvent : public vtkObject
{
  vtkTypeRevisionMacro(vtkCudaEvent, vtkObject);
public:
  static vtkCudaEvent* New();
  
  //BTX
  void Record();
  void Record(vtkCudaStream* stream);
  vtkCudaBase::State Query();
  //ETX
  void Synchronize();
  float ElapsedTime(vtkCudaEvent* otherEvent);
  
  /** @returns the Event */
  cudaEvent_t GetEvent() { return this->Event; }
  
  void PrintSelf(ostream& os, vtkIndent indent);
  
protected:
  vtkCudaEvent();
  virtual ~vtkCudaEvent();
  
  cudaEvent_t Event;
};

#endif /*VTKCUDAEVENT_H_*/
