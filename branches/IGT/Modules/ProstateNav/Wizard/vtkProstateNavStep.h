#ifndef __vtkProstateNavStep_h
#define __vtkProstateNavStep_h

#include "vtkKWWizardStep.h"
#include "vtkProstateNav.h"

class vtkProstateNavGUI;

class VTK_PROSTATENAV_EXPORT vtkProstateNavStep : public vtkKWWizardStep
{
public:
  static vtkProstateNavStep *New();
  vtkTypeRevisionMacro(vtkProstateNavStep,vtkKWWizardStep);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description: 
  // Get/Set GUI
  vtkGetObjectMacro(GUI, vtkProstateNavGUI);
  virtual void SetGUI(vtkProstateNavGUI*);

  // Description:
  // Reimplement the superclass's method (see vtkKWWizardStep).
  virtual void HideUserInterface();
  virtual void Validate();
  virtual int CanGoToSelf();

protected:
  vtkProstateNavStep();
  ~vtkProstateNavStep();

  vtkProstateNavGUI *GUI;

private:
  vtkProstateNavStep(const vtkProstateNavStep&);
  void operator=(const vtkProstateNavStep&);
};

#endif
