#ifndef __vtkProstateNavManualControlStep_h
#define __vtkProstateNavManualControlStep_h

#include "vtkProstateNavStep.h"

class VTK_PROSTATENAV_EXPORT vtkProstateNavManualControlStep : public vtkProstateNavStep
{
public:
  static vtkProstateNavManualControlStep *New();
  vtkTypeRevisionMacro(vtkProstateNavManualControlStep,vtkProstateNavStep);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Reimplement the superclass's method (see vtkKWWizardStep).
  virtual void ShowUserInterface();

protected:
  vtkProstateNavManualControlStep();
  ~vtkProstateNavManualControlStep();

private:
  vtkProstateNavManualControlStep(const vtkProstateNavManualControlStep&);
  void operator=(const vtkProstateNavManualControlStep&);
};

#endif
