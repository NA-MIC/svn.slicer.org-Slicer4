#ifndef __vtkTumorGrowthStep_h
#define __vtkTumorGrowthStep_h

#include "vtkTumorGrowth.h"
#include "vtkKWWizardStep.h"

class vtkKWFrameWithLabel; 
class vtkTumorGrowthGUI;
class vtkKWPushButton;

#define TUMORGROWTH_MENU_BUTTON_WIDTH 15
#define TUMORGROWTH_WIDGETS_LABEL_WIDTH 25
#define TUMORGROWTH_WIDGETS_SLIDER_WIDTH 100
#define TUMORGROWTH_WIDGETS_SLIDER_HEIGHT 40

class VTK_TUMORGROWTH_EXPORT vtkTumorGrowthStep : public vtkKWWizardStep
{
public:
  static vtkTumorGrowthStep *New();
  vtkTypeRevisionMacro(vtkTumorGrowthStep,vtkKWWizardStep);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description: 
  // Get/Set GUI - 
  vtkGetObjectMacro(GUI, vtkTumorGrowthGUI);
  virtual void SetGUI(vtkTumorGrowthGUI*); 

  // Description:
  // Reimplement the superclass's method (see vtkKWWizardStep).
  virtual void HideUserInterface();
  virtual void Validate();
  virtual int CanGoToSelf();

  virtual void ShowUserInterface();

  void AddGUIObservers() { } 

  virtual void TransitionCallback() { };

  // Note : This should be called directly from workflow->BackButton but it is not currently possible 
  // -> That is why we use step  
  virtual void RemoveResults() {};

  virtual void UpdateMRML() { }

  virtual void UpdateGUI() { } 
  virtual void RemoveGUIObservers() { } 

  virtual void ProcessGUIEvents(vtkObject *caller, unsigned long event, void *callData) { }

  void SetNextStep(vtkTumorGrowthStep *init) { this->NextStep = init;}

  virtual void GridCallback();

protected:
  vtkTumorGrowthStep();
  ~vtkTumorGrowthStep();

  vtkTumorGrowthGUI *GUI;
  vtkKWFrameWithLabel               *Frame;
  vtkCallbackCommand *WizardGUICallbackCommand;
  // Needed so we can clean up mess when going backwards 
  vtkTumorGrowthStep *NextStep;

  void GridRemove();
  int  GridDefine();
  void CreateGridButton(); 
  vtkKWPushButton          *GridButton;

private:
  vtkTumorGrowthStep(const vtkTumorGrowthStep&);
  void operator=(const vtkTumorGrowthStep&);
};

#endif
