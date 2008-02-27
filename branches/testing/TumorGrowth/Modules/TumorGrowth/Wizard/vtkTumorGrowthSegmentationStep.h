#ifndef __vtkTumorGrowthSegmentationStep_h
#define __vtkTumorGrowthSegmentationStep_h

#include "vtkTumorGrowthStep.h"
#include "vtkSlicerSliceLogic.h"



class vtkImageThreshold;
class vtkMRMLScalarVolumeNode;
class vtkVolumeTextureMapper3D;
class vtkPiecewiseFunction;
class vtkColorTransferFunction;
class vtkVolumeProperty;
class vtkVolume;
class vtkMatrix4x4;         
class vtkKWFrame;
class vtkKWLabel;
class vtkKWRange;
class vtkKWScaleWithEntry;

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
  virtual void ThresholdRangeChangedCallback(double min, double max);
  virtual void TransitionCallback(); 
  // We call this function in order to remove nodes when going backwards 
  virtual void RemoveResults()  { 
    this->PreSegmentScan1Remove();
    this->SliceLogicRemove();
  }

  vtkGetObjectMacro(PreSegment,vtkImageThreshold);

  // Description:
  // accessor
  vtkGetObjectMacro (SliceLogic, vtkSlicerSliceLogic);

  void ProcessGUIEvents(vtkObject *caller, unsigned long event, void *callData);

protected:
  vtkTumorGrowthSegmentationStep();
  ~vtkTumorGrowthSegmentationStep();

  static void WizardGUICallback(vtkObject *caller, unsigned long event, void *clientData, void *callData );


  vtkKWFrame *ThresholdFrame;
  vtkKWRange *ThresholdRange;
  vtkKWLabel *ThresholdLabel;

private:
  vtkTumorGrowthSegmentationStep(const vtkTumorGrowthSegmentationStep&);
  void operator=(const vtkTumorGrowthSegmentationStep&);

  void PreSegmentScan1Remove();
  void PreSegmentScan1Define();

  void SegmentScan1Remove();
  int SegmentScan1Define();

  void SliceLogicRemove();
  void SliceLogicDefine();
  
  void SetPreSegment_Render_BandPassFilter(double min, double max);

  vtkImageThreshold *PreSegment;
  vtkMRMLScalarVolumeNode *PreSegmentNode; 
  vtkMRMLScalarVolumeNode *SegmentNode; 
  vtkVolumeTextureMapper3D *PreSegment_Render_Mapper;
  vtkPiecewiseFunction     *PreSegment_Render_BandPassFilter;
  vtkColorTransferFunction *PreSegment_Render_ColorMapping;
  vtkVolumeProperty        *PreSegment_Render_VolumeProperty;
  vtkVolume                *PreSegment_Render_Volume;
  vtkMatrix4x4             *PreSegment_Render_OrientationMatrix; 

  vtkSlicerSliceLogic *SliceLogic;
  vtkKWScaleWithEntry *SliceController_OffsetScale;
};

#endif
