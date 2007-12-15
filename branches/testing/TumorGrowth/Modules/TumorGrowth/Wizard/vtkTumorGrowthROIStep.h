#ifndef __vtkTumorGrowthROIStep_h
#define __vtkTumorGrowthROIStep_h

#include "vtkTumorGrowthStep.h"

class vtkKWFrame;
class vtkKWMatrixWidgetWithLabel;
class vtkKWPushButton;
class vtkSlicerModuleCollapsibleFrame;
class vtkSlicerSliceGUI;
class vtkRenderWindowInteractor;
class vtkMRMLScalarVolumeNode;

class VTK_TUMORGROWTH_EXPORT vtkTumorGrowthROIStep : public vtkTumorGrowthStep
{
public:
  static vtkTumorGrowthROIStep *New();
  vtkTypeRevisionMacro(vtkTumorGrowthROIStep,vtkTumorGrowthStep);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Reimplement the superclass's method (see vtkKWWizardStep).
  virtual void ShowUserInterface();
  virtual void HideUserInterface();

  // Description:
  // Callbacks.
  void ROIMaxChangedCallback(int row, int col, const char *value); 
  void ROIMinChangedCallback(int row, int col, const char *value); 
  virtual void TransitionCallback(); 

  // Description:
  // Callback functions for buttons
  void ProcessGUIEvents(vtkObject *caller, unsigned long event, void *callData);
  void AddGUIObservers();
  void RemoveGUIObservers();
 
protected:
  vtkTumorGrowthROIStep();
  ~vtkTumorGrowthROIStep();

  vtkKWFrame                        *FrameButtons;
  vtkKWFrame                        *FrameBlank;
  vtkSlicerModuleCollapsibleFrame   *FrameROI;
  vtkKWPushButton           *ButtonsShow;
  vtkKWPushButton           *ButtonsReset;
  vtkKWMatrixWidgetWithLabel *ROIMinVector;
  vtkKWMatrixWidgetWithLabel *ROIMaxVector;
  static void WizardGUICallback(vtkObject *caller, unsigned long event, void *clientData, void *callData );
 
private:
  vtkTumorGrowthROIStep(const vtkTumorGrowthROIStep&);
  void operator=(const vtkTumorGrowthROIStep&);
 
  void ROIReset();
  void ROIUpdateWithNewSample(int ijkSample[3]);
  void ROIUpdateWithNode();
  int  ROICheck();

  int ROIMapShow();
  void ROIMapRemove();

  void AddROISamplingGUIObservers();
  void RemoveROISamplingGUIObservers();
 
  void RetrieveInteractorIJKCoordinates(vtkSlicerSliceGUI *sliceGUI, vtkRenderWindowInteractor *rwi,int coords[3]);

  vtkMRMLScalarVolumeNode *ROILabelMapNode;
};

#endif
