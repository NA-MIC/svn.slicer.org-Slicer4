#ifndef __vtkProstateNavStep_h
#define __vtkProstateNavStep_h

#include "vtkKWWizardStep.h"
#include "vtkProstateNav.h"
#include "vtkCommand.h"

class vtkProstateNavGUI;
class vtkProstateNavLogic;

class VTK_PROSTATENAV_EXPORT vtkProstateNavStep : public vtkKWWizardStep
{
public:
  static vtkProstateNavStep *New();
  vtkTypeRevisionMacro(vtkProstateNavStep,vtkKWWizardStep);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description: 
  // Get/Set GUI
  vtkGetObjectMacro(GUI, vtkProstateNavGUI);
  vtkGetObjectMacro(Logic, vtkProstateNavLogic);
  virtual void SetGUI(vtkProstateNavGUI*);
  virtual void SetLogic(vtkProstateNavLogic*);

  vtkGetMacro(InGUICallbackFlag, int);
  void SetInGUICallbackFlag (int flag) {
    this->InGUICallbackFlag = flag;
  };
  void SetTitleBackgroundColor (double r, double g, double b) {
    this->TitleBackgroundColor[0] = r;
    this->TitleBackgroundColor[1] = g;
    this->TitleBackgroundColor[2] = b;
  };

  // Description:
  // Reimplement the superclass's method (see vtkKWWizardStep).
  virtual void HideUserInterface();
  virtual void Validate();
  virtual int CanGoToSelf();
  virtual void ShowUserInterface();
  virtual void ProcessGUIEvents(vtkObject *caller, unsigned long event, void *callData) {};

protected:
  vtkProstateNavStep();
  ~vtkProstateNavStep();

  static void GUICallback( vtkObject *caller,
                           unsigned long eid, void *clientData, void *callData );
  
  int InGUICallbackFlag;

  vtkProstateNavGUI   *GUI;
  vtkProstateNavLogic *Logic;
  vtkCallbackCommand  *GUICallbackCommand;

  double TitleBackgroundColor[3];

private:
  vtkProstateNavStep(const vtkProstateNavStep&);
  void operator=(const vtkProstateNavStep&);

};

#endif
