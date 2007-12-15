#ifndef __vtkTumorGrowthSegmentationStep_h
#define __vtkTumorGrowthSegmentationStep_h

#include "vtkTumorGrowthStep.h"

class vtkKWThumbWheel;
class vtkImageThreshold;
class vtkMRMLScalarVolumeNode;

class VTK_TUMORGROWTH_EXPORT vtkTumorGrowthSegmentationStep : public vtkTumorGrowthStep
{
public:
  static vtkTumorGrowthSegmentationStep *New();
  vtkTypeRevisionMacro(vtkTumorGrowthSegmentationStep,vtkTumorGrowthStep);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Reimplement the superclass's method (see vtkKWWizardStep).
  virtual void ShowUserInterface();

  // Description:
  // Callbacks.
  virtual void ThresholdChangedCallback(double value);
  virtual void TransitionCallback(); 

protected:
  vtkTumorGrowthSegmentationStep();
  ~vtkTumorGrowthSegmentationStep();

  vtkKWThumbWheel          *ThresholdScale;

private:
  vtkTumorGrowthSegmentationStep(const vtkTumorGrowthSegmentationStep&);
  void operator=(const vtkTumorGrowthSegmentationStep&);

  void PreSegmentScan1Remove();
  void PreSegmentScan1Define();

  void SegmentScan1Remove();
  int SegmentScan1Define();
  
  vtkImageThreshold *PreSegment;
  vtkMRMLScalarVolumeNode *PreSegmentNode; 
  vtkMRMLScalarVolumeNode *SegmentNode; 
};

#endif
