#ifndef __vtkTumorGrowthStep_h
#define __vtkTumorGrowthStep_h

#include "vtkTumorGrowth.h"
#include "vtkKWWizardStep.h"

class vtkKWFrameWithLabel; 
class vtkTumorGrowthGUI;

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

  virtual void UpdateMRML() { }

  virtual void UpdateGUI() { } 
  virtual void RemoveGUIObservers() { } 

  virtual void ProcessGUIEvents(vtkObject *caller, unsigned long event, void *callData) { }

protected:
  vtkTumorGrowthStep();
  ~vtkTumorGrowthStep();

  vtkTumorGrowthGUI *GUI;
  vtkKWFrameWithLabel               *Frame;
  
private:
  vtkTumorGrowthStep(const vtkTumorGrowthStep&);
  void operator=(const vtkTumorGrowthStep&);
};

#endif
