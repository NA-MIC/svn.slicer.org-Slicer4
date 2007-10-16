#ifndef __vtkProstateNavCalibrationStep_h
#define __vtkProstateNavCalibrationStep_h

#include "vtkProstateNavStep.h"

class VTK_PROSTATENAV_EXPORT vtkProstateNavCalibrationStep : public vtkProstateNavStep
{
public:
  static vtkProstateNavCalibrationStep *New();
  vtkTypeRevisionMacro(vtkProstateNavCalibrationStep,vtkProstateNavStep);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Reimplement the superclass's method (see vtkKWWizardStep).
  virtual void ShowUserInterface();

protected:
  vtkProstateNavCalibrationStep();
  ~vtkProstateNavCalibrationStep();

private:
  vtkProstateNavCalibrationStep(const vtkProstateNavCalibrationStep&);
  void operator=(const vtkProstateNavCalibrationStep&);
};

#endif
