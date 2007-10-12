#ifndef __vtkProstateNavConfigFileStep_h
#define __vtkProstateNavConfigFileStep_h

#include "vtkProstateNavStep.h"

class vtkKWLoadSaveButtonWithLabel;
class vtkKWFrame;
class vtkKWEntry;
class vtkKWCheckButton;

class VTK_PROSTATENAV_EXPORT vtkProstateNavConfigFileStep :
  public vtkProstateNavStep
{
public:
  static vtkProstateNavConfigFileStep *New();
  vtkTypeRevisionMacro(vtkProstateNavConfigFileStep,vtkProstateNavStep);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Reimplement the superclass's method (see vtkKWWizardStep).
  virtual void ShowUserInterface();

protected:
  vtkProstateNavConfigFileStep();
  ~vtkProstateNavConfigFileStep();

  vtkKWFrame *ConfigNTFrame;
  vtkKWFrame *ConnectNTFrame;
  vtkKWLoadSaveButtonWithLabel *LoadConfigButtonNT;
  vtkKWEntry *ConfigFileEntryNT;
  vtkKWCheckButton *ConnectCheckButtonNT;

private:
  vtkProstateNavConfigFileStep(const vtkProstateNavConfigFileStep&);
  void operator=(const vtkProstateNavConfigFileStep&);
};

#endif
