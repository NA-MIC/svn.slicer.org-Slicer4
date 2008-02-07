#ifndef __vtkTumorGrowthTypeStep_h
#define __vtkTumorGrowthTypeStep_h

#include "vtkTumorGrowthStep.h"

class vtkKWCheckButton;
class vtkKWFrame;

class VTK_TUMORGROWTH_EXPORT vtkTumorGrowthTypeStep : public vtkTumorGrowthStep
{
public:
  static vtkTumorGrowthTypeStep *New();
  vtkTypeRevisionMacro(vtkTumorGrowthTypeStep,vtkTumorGrowthStep);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Reimplement the superclass's method (see vtkKWWizardStep).
  virtual void ShowUserInterface();

 // Description:
  // Callback functions for buttons
  void ProcessGUIEvents(vtkObject *caller, unsigned long event, void *callData);
  void AddGUIObservers();
  void RemoveGUIObservers();

  virtual void TransitionCallback(); 

protected:
  vtkTumorGrowthTypeStep();
  ~vtkTumorGrowthTypeStep();

  static void WizardGUICallback(vtkObject *caller, unsigned long event, void *clientData, void *callData );
  void RemoveAnalysisOutput();

  vtkKWFrame*FrameTypeIntensity;
  vtkKWFrame*FrameTypeSegmentation;
  vtkKWFrame*FrameTypeJacobian;

  vtkKWCheckButton *TypeIntensityCheckButton;
  vtkKWCheckButton *TypeSegmentationCheckButton; 
  vtkKWCheckButton *TypeJacobianCheckButton; 


private:
  vtkTumorGrowthTypeStep(const vtkTumorGrowthTypeStep&);
  void operator=(const vtkTumorGrowthTypeStep&);
};

#endif
