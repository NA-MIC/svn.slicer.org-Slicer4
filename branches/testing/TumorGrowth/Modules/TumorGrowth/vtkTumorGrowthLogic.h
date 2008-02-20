#ifndef __vtkTumorGrowthLogic_h
#define __vtkTumorGrowthLogic_h

#include "vtkSlicerModuleLogic.h"
#include "vtkTumorGrowth.h"
#include "vtkMRMLTumorGrowthNode.h"
#include "vtkGeneralTransform.h"

#include <string>
#include <map>

class vtkMRMLScene;
class vtkMRMLScalarNode;
class vtkMRMLVolumeNode;
class vtkSlicerApplication;
class vtkImageMathematics;

class VTK_TUMORGROWTH_EXPORT vtkTumorGrowthLogic : 
  public vtkSlicerModuleLogic
{
public:
  static vtkTumorGrowthLogic *New();
  vtkTypeMacro(vtkTumorGrowthLogic,vtkSlicerModuleLogic);
  void PrintSelf(ostream& os, vtkIndent indent);

 // TODO: do we need to observe MRML here?
 // pohl: I so not I follow example vtkGradnientAnisotrpoicDiffusionoFilterGUI
 // virtual void ProcessMrmlEvents ( vtkObject *caller, unsigned long event,
 //                                  void *callData ){};
 // void ProcessMRMLEvents(vtkObject* caller, unsigned long event, void* callData);



  // Description: The name of the Module---this is used to construct
  // the proc invocations
  vtkGetStringMacro (ModuleName);
  vtkSetStringMacro (ModuleName);

  vtkGetObjectMacro (GlobalTransform, vtkGeneralTransform);
  vtkGeneralTransform* CreateGlobalTransform();
  vtkGetObjectMacro (LocalTransform,  vtkGeneralTransform);
  vtkGeneralTransform* CreateLocalTransform();

  vtkGetObjectMacro (TumorGrowthNode, vtkMRMLTumorGrowthNode);
  void SetAndObserveTumorGrowthNode(vtkMRMLTumorGrowthNode *n) 
    {
    vtkSetAndObserveMRMLNodeMacro( this->TumorGrowthNode, n);
    }


  //
  // progress bar related functions: not currently used, likely to
  // change
  vtkGetStringMacro(ProgressCurrentAction);
  vtkGetMacro(ProgressGlobalFractionCompleted, double);
  vtkGetMacro(ProgressCurrentFractionCompleted, double);

  void RegisterMRMLNodesWithScene(); 

  // special testing functions
  vtkMRMLScalarVolumeNode* CreateSuperSample(int ScanNum, vtkSlicerApplication *application);

  int CheckROI(vtkMRMLVolumeNode* volumeNode);

  // copied from vtkMRMLScalarVolumeNode* vtkSlicerVolumesLogic::CloneVolume without deep copy
  vtkMRMLScalarVolumeNode* CreateVolumeNode(vtkMRMLVolumeNode *volumeNode, char *name);

  // Main Growth Function 
  int AnalyzeGrowth(vtkSlicerApplication *application);
  double MeassureGrowth(vtkSlicerApplication *app);
  void DeleteAnalyzeOutput(vtkSlicerApplication *app);



  vtkImageThreshold* CreateAnalysis_Intensity_Final();
  vtkImageThreshold* CreateAnalysis_Intensity_ROINegativeBin();
  vtkImageThreshold* CreateAnalysis_Intensity_ROIPositiveBin();
  vtkImageMathematics* CreateAnalysis_Intensity_ROIBinReal();

  vtkImageSumOverVoxels* CreateAnalysis_Intensity_ROITotal();

  vtkSetMacro(Analysis_Intensity_Mean,double);
  vtkGetMacro(Analysis_Intensity_Mean,double);
  vtkSetMacro(Analysis_Intensity_Variance,double);
  vtkGetMacro(Analysis_Intensity_Variance,double);
  vtkSetMacro(Analysis_Intensity_Threshold,double);
  vtkGetMacro(Analysis_Intensity_Threshold,double);

  vtkSetMacro(SaveVolumeFlag,int);
  vtkGetMacro(SaveVolumeFlag,int);

  void SaveVolume(vtkSlicerApplication *app, vtkMRMLVolumeNode *volNode);
  // Save the output ignoring SaveVolumeFlag
  void SaveVolumeForce(vtkSlicerApplication *app, vtkMRMLVolumeNode *volNode);

  void SaveVolumeFileName(vtkMRMLVolumeNode *volNode, char* FileName);

  vtkMRMLVolumeNode* LoadVolume(vtkSlicerApplication *app, char* fileName, int LabelMapFlag,char* volumeName);

  void PrintResult(ostream& os, vtkSlicerApplication *app);

  void PrintText(char *TEXT);

private:
  vtkTumorGrowthLogic();
  ~vtkTumorGrowthLogic();
  vtkTumorGrowthLogic(const vtkTumorGrowthLogic&);
  void operator=(const vtkTumorGrowthLogic&);

  // not currently used
  vtkSetStringMacro(ProgressCurrentAction);
  vtkSetMacro(ProgressGlobalFractionCompleted, double);
  vtkSetMacro(ProgressCurrentFractionCompleted, double);

  void SourceAnalyzeTclScripts(vtkSlicerApplication *app);


  //
  // because the mrml nodes are very complicated for this module, we
  // delegate the handeling of them to a MRML manager
  vtkMRMLTumorGrowthNode* TumorGrowthNode;

  char *ModuleName;

  //
  // information related to progress bars: this mechanism is not
  // currently implemented and might me best implemented elsewhere
  char*  ProgressCurrentAction;
  double ProgressGlobalFractionCompleted;
  double ProgressCurrentFractionCompleted;

  vtkGeneralTransform* GlobalTransform; 
  vtkGeneralTransform* LocalTransform; 

  double Analysis_Intensity_Mean;
  double Analysis_Intensity_Variance;
  double Analysis_Intensity_Threshold;

  vtkImageThreshold     *Analysis_Intensity_Final;
  vtkImageThreshold     *Analysis_Intensity_ROINegativeBin;
  vtkImageThreshold     *Analysis_Intensity_ROIPositiveBin;
  vtkImageMathematics   *Analysis_Intensity_ROIBinReal;
  vtkImageSumOverVoxels *Analysis_Intensity_ROITotal;

  int SaveVolumeFlag;
};

#endif
