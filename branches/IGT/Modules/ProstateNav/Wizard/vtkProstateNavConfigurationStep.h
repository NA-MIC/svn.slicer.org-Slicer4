#ifndef __vtkProstateNavConfigurationStep_h
#define __vtkProstateNavConfigurationStep_h

#include "vtkProstateNavStep.h"

class vtkKWLoadSaveButtonWithLabel;
class vtkKWFrame;
class vtkKWEntry;
class vtkKWCheckButton;

class VTK_PROSTATENAV_EXPORT vtkProstateNavConfigurationStep :
  public vtkProstateNavStep
{
public:
  static vtkProstateNavConfigurationStep *New();
  vtkTypeRevisionMacro(vtkProstateNavConfigurationStep,vtkProstateNavStep);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Reimplement the superclass's method (see vtkKWWizardStep).
  virtual void ShowUserInterface();

protected:
  vtkProstateNavConfigurationStep();
  ~vtkProstateNavConfigurationStep();

  vtkKWFrame *ConfigNTFrame;
  vtkKWFrame *ConnectNTFrame;
  vtkKWLoadSaveButtonWithLabel *LoadConfigButtonNT;
  vtkKWEntry *ConfigFileEntryNT;
  vtkKWCheckButton *ConnectCheckButtonNT;

private:
  vtkProstateNavConfigurationStep(const vtkProstateNavConfigurationStep&);
  void operator=(const vtkProstateNavConfigurationStep&);
};

#endif
