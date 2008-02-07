#ifndef __vtkTumorGrowthAnalysisStep_h
#define __vtkTumorGrowthAnalysisStep_h

#include "vtkTumorGrowthStep.h"

class vtkKWThumbWheel;
class vtkKWLabel;
class vtkKWFrameWithLabel; 
class vtkKWPushButton;

class VTK_TUMORGROWTH_EXPORT vtkTumorGrowthAnalysisStep : public vtkTumorGrowthStep
{
public:
  static vtkTumorGrowthAnalysisStep *New();
  vtkTypeRevisionMacro(vtkTumorGrowthAnalysisStep,vtkTumorGrowthStep);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Reimplement the superclass's method (see vtkKWWizardStep).
  virtual void ShowUserInterface();

  // Description:
  // Callbacks.
  virtual void SensitivityChangedCallback(double value);

  void ResetPipelineCallback();
  // Kilian Work here tomorrow 
  virtual void TransitionCallback() { }; 
  // We call this function in order to remove nodes when going backwards 
  virtual void RemoveResults(); 

  void ProcessGUIEvents(vtkObject *caller, unsigned long event, void *callData);
  void RemoveGUIObservers();
  void AddGUIObservers(); 

protected:
  vtkTumorGrowthAnalysisStep();
  ~vtkTumorGrowthAnalysisStep();

  // vtkKWScaleWithEntry      *SensitivetyScale;
  vtkKWThumbWheel            *SensitivityScale;
  vtkKWLabel                 *GrowthLabel;

  vtkKWFrameWithLabel       *FrameButtons;
  vtkKWPushButton           *ButtonsSnapshot;
  vtkKWPushButton           *ButtonsSave;

  static void WizardGUICallback(vtkObject *caller, unsigned long event, void *clientData, void *callData );


private:
  vtkTumorGrowthAnalysisStep(const vtkTumorGrowthAnalysisStep&);
  void operator=(const vtkTumorGrowthAnalysisStep&);
};

#endif


