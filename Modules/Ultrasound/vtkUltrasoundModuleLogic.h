#ifndef __vtkUltrasoundModuleLogic_h
#define __vtkUltrasoundModuleLogic_h
#include "vtkSlicerModuleLogic.h"
#include "vtkUltrasoundModule.h"

class VTK_ULTRASOUNDMODULE_EXPORT vtkUltrasoundModuleLogic :public vtkSlicerModuleLogic
{
public:
  static vtkUltrasoundModuleLogic *New();
  vtkTypeMacro(vtkUltrasoundModuleLogic, vtkSlicerModuleLogic);
  void PrintSelf(ostream& os, vtkIndent indent);

  // TODO: do we need to observe MRML here?
  //virtual void ProcessMrmlEvents ( vtkObject *caller, unsigned long event,
    //                               void *callData ){};

  


protected:
  vtkUltrasoundModuleLogic();
  ~vtkUltrasoundModuleLogic();
  vtkUltrasoundModuleLogic(const vtkUltrasoundModuleLogic&);
  void operator=(const vtkUltrasoundModuleLogic&);
};

#endif
