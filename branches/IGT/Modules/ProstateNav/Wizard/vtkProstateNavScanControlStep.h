#ifndef __vtkProstateNavScanControlStep_h
#define __vtkProstateNavScanControlStep_h

#include "vtkProstateNavStep.h"

class VTK_PROSTATENAV_EXPORT vtkProstateNavScanControlStep : public vtkProstateNavStep
{
public:
  static vtkProstateNavScanControlStep *New();
  vtkTypeRevisionMacro(vtkProstateNavScanControlStep,vtkProstateNavStep);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Reimplement the superclass's method (see vtkKWWizardStep).
  virtual void ShowUserInterface();

protected:
  vtkProstateNavScanControlStep();
  ~vtkProstateNavScanControlStep();

private:
  vtkProstateNavScanControlStep(const vtkProstateNavScanControlStep&);
  void operator=(const vtkProstateNavScanControlStep&);
};

#endif
