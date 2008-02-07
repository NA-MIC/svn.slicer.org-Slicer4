#ifndef __vtkTumorGrowthSegmentationStep_h
#define __vtkTumorGrowthSegmentationStep_h

#include "vtkTumorGrowthStep.h"

class vtkKWThumbWheel;
class vtkImageThreshold;
class vtkMRMLScalarVolumeNode;
class vtkVolumeTextureMapper3D;
class vtkPiecewiseFunction;
class vtkColorTransferFunction;
class vtkVolumeProperty;
class vtkVolume;
class vtkMatrix4x4;         

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
  // We call this function in order to remove nodes when going backwards 
  virtual void RemoveResults()  { this->PreSegmentScan1Remove();}

  vtkGetObjectMacro(PreSegment,vtkImageThreshold);

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
  
  void SetPreSegment_Render_BandPassFilter(double value);

  vtkImageThreshold *PreSegment;
  vtkMRMLScalarVolumeNode *PreSegmentNode; 
  vtkMRMLScalarVolumeNode *SegmentNode; 
  vtkVolumeTextureMapper3D *PreSegment_Render_Mapper;
  vtkPiecewiseFunction     *PreSegment_Render_BandPassFilter;
  vtkColorTransferFunction *PreSegment_Render_ColorMapping;
  vtkVolumeProperty        *PreSegment_Render_VolumeProperty;
  vtkVolume                *PreSegment_Render_Volume;
  vtkMatrix4x4             *PreSegment_Render_OrientationMatrix; 
};

#endif
