#ifndef __vtkTumorGrowthAnalysisStep_h
#define __vtkTumorGrowthAnalysisStep_h

#include "vtkTumorGrowthStep.h"

class vtkKWThumbWheel;
class vtkKWLabel;

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

protected:
  vtkTumorGrowthAnalysisStep();
  ~vtkTumorGrowthAnalysisStep();

  // vtkKWScaleWithEntry      *SensitivetyScale;
  vtkKWThumbWheel            *SensitivityScale;
  vtkKWLabel                 *GrowthLabel;

private:
  vtkTumorGrowthAnalysisStep(const vtkTumorGrowthAnalysisStep&);
  void operator=(const vtkTumorGrowthAnalysisStep&);
};

#endif
