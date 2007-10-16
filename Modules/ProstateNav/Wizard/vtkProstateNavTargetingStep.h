#ifndef __vtkProstateNavTargetingStep_h
#define __vtkProstateNavTargetingStep_h

#include "vtkProstateNavStep.h"

class VTK_PROSTATENAV_EXPORT vtkProstateNavTargetingStep : public vtkProstateNavStep
{
public:
  static vtkProstateNavTargetingStep *New();
  vtkTypeRevisionMacro(vtkProstateNavTargetingStep,vtkProstateNavStep);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Reimplement the superclass's method (see vtkKWWizardStep).
  virtual void ShowUserInterface();

protected:
  vtkProstateNavTargetingStep();
  ~vtkProstateNavTargetingStep();

private:
  vtkProstateNavTargetingStep(const vtkProstateNavTargetingStep&);
  void operator=(const vtkProstateNavTargetingStep&);
};

#endif
