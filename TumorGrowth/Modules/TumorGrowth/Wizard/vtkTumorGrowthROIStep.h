#ifndef __vtkTumorGrowthROIStep_h
#define __vtkTumorGrowthROIStep_h

#include "vtkTumorGrowthStep.h"

class vtkKWFrame;
class vtkKWMatrixWidgetWithLabel;
class vtkKWPushButton;
class vtkSlicerModuleCollapsibleFrame;

class VTK_TUMORGROWTH_EXPORT vtkTumorGrowthROIStep : public vtkTumorGrowthStep
{
public:
  static vtkTumorGrowthROIStep *New();
  vtkTypeRevisionMacro(vtkTumorGrowthROIStep,vtkTumorGrowthStep);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Reimplement the superclass's method (see vtkKWWizardStep).
  virtual void ShowUserInterface();

  // Description:
  // Callbacks.
  void ROIMaxChangedCallback(vtkIdType sel_vol_id, double value);
  void ROIMinChangedCallback(vtkIdType sel_vol_id, double value);
  virtual void TransitionCallback(); 
  
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
 
private:
  vtkTumorGrowthROIStep(const vtkTumorGrowthROIStep&);
  void operator=(const vtkTumorGrowthROIStep&);
};

#endif
