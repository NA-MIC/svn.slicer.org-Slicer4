#include <string>
#include <iostream>
#include <sstream>

#include "vtkObjectFactory.h"
#include "vtkTumorGrowthLogic.h"
#include "vtkTumorGrowth.h"
#include "vtkMRMLScene.h"
#include "vtkMRMLTumorGrowthNode.h"
#include "vtkImageClip.h"
#include "vtkImageChangeInformation.h"
#include "vtkImageResample.h"
#include "vtkSlicerApplication.h"
#include "vtkImageMathematics.h"
#include "vtkImageSumOverVoxels.h"

#include "vtkSlicerVolumesLogic.h"
#include "vtkSlicerVolumesGUI.h"
//#include "vtkSlicerApplication.h"

#define ERROR_NODE_VTKID 0

//----------------------------------------------------------------------------
vtkTumorGrowthLogic* vtkTumorGrowthLogic::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = 
    vtkObjectFactory::CreateInstance("vtkTumorGrowthLogic");
  if(ret)
    {
    return (vtkTumorGrowthLogic*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkTumorGrowthLogic;
}


//----------------------------------------------------------------------------
vtkTumorGrowthLogic::vtkTumorGrowthLogic()
{
  this->ModuleName = NULL;

  this->ProgressCurrentAction = NULL;
  this->ProgressGlobalFractionCompleted = 0.0;
  this->ProgressCurrentFractionCompleted = 0.0;

  //this->DebugOn();
  this->TumorGrowthNode = NULL; 
  this->LocalTransform = NULL; 
  this->GlobalTransform = NULL; 

  this->Analysis_Intensity_Mean      = 0.0;
  this->Analysis_Intensity_Variance  = 0.0;
  this->Analysis_Intensity_Threshold = 0.0;

  this->Analysis_Intensity_Final          = NULL;
  this->Analysis_Intensity_ROINegativeBin = NULL;
  this->Analysis_Intensity_ROIPositiveBin = NULL;
  this->Analysis_Intensity_ROIBinReal     = NULL;
  this->Analysis_Intensity_ROITotal       = NULL;

  // if set to zero then SaveVolume will not do anything 
  this->SaveVolumeFlag = 0;  
}


//----------------------------------------------------------------------------
vtkTumorGrowthLogic::~vtkTumorGrowthLogic()
{
  vtkSetMRMLNodeMacro(this->TumorGrowthNode, NULL);
  this->SetProgressCurrentAction(NULL);
  this->SetModuleName(NULL);

  if (this->LocalTransform) {
    this->LocalTransform->Delete();
    this->LocalTransform = NULL;
  }

  if (this->GlobalTransform) {
    this->LocalTransform->Delete();
    this->LocalTransform = NULL;
  }
  
  if (this->Analysis_Intensity_Final) {
    this->Analysis_Intensity_Final->Delete();
    this->Analysis_Intensity_Final= NULL;
  }

  if (this->Analysis_Intensity_ROINegativeBin) {
    this->Analysis_Intensity_ROINegativeBin->Delete();
    this->Analysis_Intensity_ROINegativeBin = NULL;
  }

  if (this->Analysis_Intensity_ROIPositiveBin) {
    this->Analysis_Intensity_ROIPositiveBin->Delete();
    this->Analysis_Intensity_ROIPositiveBin = NULL;
  }

  if (this->Analysis_Intensity_ROIBinReal) {
    this->Analysis_Intensity_ROIBinReal->Delete();
    this->Analysis_Intensity_ROIBinReal = NULL;
  }

  if (this->Analysis_Intensity_ROITotal) {
    this->Analysis_Intensity_ROITotal->Delete();
    this->Analysis_Intensity_ROITotal = NULL;
  }

}

//----------------------------------------------------------------------------
void vtkTumorGrowthLogic::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
  // !!! todo
}

vtkGeneralTransform* vtkTumorGrowthLogic::CreateGlobalTransform() 
{
  this->GlobalTransform = vtkGeneralTransform::New();
  return this->GlobalTransform;
}

vtkGeneralTransform* vtkTumorGrowthLogic::CreateLocalTransform() 
{
  this->LocalTransform = vtkGeneralTransform::New();
  return this->LocalTransform;
}

int vtkTumorGrowthLogic::CheckROI(vtkMRMLVolumeNode* volumeNode) {
  if (!volumeNode || !this->TumorGrowthNode) return 0;

  int* dimensions = volumeNode->GetImageData()->GetDimensions();

  for (int i = 0 ; i < 3 ; i++) {
    if ((this->TumorGrowthNode->GetROIMax(i) < 0) || (this->TumorGrowthNode->GetROIMax(i) >= dimensions[i])) return 0 ;
    if ((this->TumorGrowthNode->GetROIMin(i) < 0) || (this->TumorGrowthNode->GetROIMin(i) >= dimensions[i])) return 0 ;
    if (this->TumorGrowthNode->GetROIMax(i) < this->TumorGrowthNode->GetROIMin(i)) return 0;
  }
  return 1;
}


// Give another Volume as an example - creates a volume without anything in it 
// copied from vtkMRMLScalarVolumeNode* vtkSlicerVolumesLogic::CloneVolume without deep copy
vtkMRMLScalarVolumeNode* vtkTumorGrowthLogic::CreateVolumeNode(vtkMRMLVolumeNode *volumeNode, char *name) {
  if (!this->TumorGrowthNode || !volumeNode ) 
    {
    return NULL;
    }

  // clone the display node
  vtkMRMLScalarVolumeDisplayNode *clonedDisplayNode = vtkMRMLScalarVolumeDisplayNode::New();
  clonedDisplayNode->CopyWithScene(volumeNode->GetDisplayNode());
  this->TumorGrowthNode->GetScene()->AddNode(clonedDisplayNode);

  // clone the volume node
  vtkMRMLScalarVolumeNode *clonedVolumeNode = vtkMRMLScalarVolumeNode::New();
  clonedVolumeNode->CopyWithScene(volumeNode);
  clonedVolumeNode->SetStorageNodeID(NULL);
  clonedVolumeNode->SetName(name);
  clonedVolumeNode->SetAndObserveDisplayNodeID(clonedDisplayNode->GetID());

  if (0) {
    // copy over the volume's data
    vtkImageData* clonedVolumeData = vtkImageData::New(); 
    clonedVolumeData->DeepCopy(volumeNode->GetImageData());
    clonedVolumeNode->SetAndObserveImageData( clonedVolumeData );
    clonedVolumeNode->SetModifiedSinceRead(1);
    clonedVolumeData->Delete();
  }

  // add the cloned volume to the scene
  this->TumorGrowthNode->GetScene()->AddNode(clonedVolumeNode);

  // remove references
  clonedVolumeNode->Delete();
  clonedDisplayNode->Delete();

  return (clonedVolumeNode);
 
}

vtkMRMLScalarVolumeNode* vtkTumorGrowthLogic::CreateSuperSample(int ScanNum,  vtkSlicerApplication *application) {
  // ---------------------------------
  // Initialize Variables 
  if (!this->TumorGrowthNode)  return NULL;

  vtkMRMLVolumeNode* volumeNode = NULL;

  if (ScanNum > 1)  {
    volumeNode = vtkMRMLVolumeNode::SafeDownCast(this->TumorGrowthNode->GetScene()->GetNodeByID(this->TumorGrowthNode->GetScan2_GlobalRef()));
  } else {
    volumeNode = vtkMRMLVolumeNode::SafeDownCast(this->TumorGrowthNode->GetScene()->GetNodeByID(this->TumorGrowthNode->GetScan1_Ref()));
  }

  if (!this->CheckROI(volumeNode)) return NULL;

  double SuperSampleSpacing = -1;
  
  // ---------------------------------
  // Perform Super Sampling 

  vtkImageClip  *ROI = vtkImageClip::New();
     ROI->SetInput(volumeNode->GetImageData());
     ROI->SetOutputWholeExtent(this->TumorGrowthNode->GetROIMin(0),this->TumorGrowthNode->GetROIMax(0),
                   this->TumorGrowthNode->GetROIMin(1),this->TumorGrowthNode->GetROIMax(1),
                   this->TumorGrowthNode->GetROIMin(2),this->TumorGrowthNode->GetROIMax(2)); 
     ROI->ClipDataOn();   
     ROI->Update(); 

  vtkImageChangeInformation *ROIExtent = vtkImageChangeInformation::New();
     ROIExtent->SetInput(ROI->GetOutput());
     ROIExtent->SetOutputExtentStart(0,0,0); 
  ROIExtent->Update();
 
  // In old version we saved the file here 

  // Determine Coeficients for resampling   
  double *Spacing = volumeNode->GetSpacing();
  if (ScanNum == 1) {
    int size = this->TumorGrowthNode->GetROIMax(0) - this->TumorGrowthNode->GetROIMin(0) + 1;
    double TempSpacing = double(size) * Spacing[0] / 100.0;
    SuperSampleSpacing = (TempSpacing < 0.3 ?  0.3 : TempSpacing);
    
    size = this->TumorGrowthNode->GetROIMax(1) - this->TumorGrowthNode->GetROIMin(1) + 1;
    TempSpacing = double(size) * Spacing[1] / 100.0;
    if (TempSpacing > SuperSampleSpacing) { SuperSampleSpacing = TempSpacing;}

    size = this->TumorGrowthNode->GetROIMax(2) - this->TumorGrowthNode->GetROIMin(2) + 1;
    TempSpacing = double(size) * Spacing[2] / 100.0;
    if (TempSpacing > SuperSampleSpacing) { SuperSampleSpacing = TempSpacing;}
    
    this->TumorGrowthNode->SetSuperSampled_Spacing(SuperSampleSpacing);
    double SuperSampleVol = SuperSampleSpacing*SuperSampleSpacing*SuperSampleSpacing;
    this->TumorGrowthNode->SetSuperSampled_VoxelVolume(SuperSampleVol); 
    this->TumorGrowthNode->SetSuperSampled_RatioNewOldSpacing(SuperSampleVol/(Spacing[0]*Spacing[1]*Spacing[2]));
    this->TumorGrowthNode->SetScan1_VoxelVolume(Spacing[0]*Spacing[1]*Spacing[2]);

  } else {
    SuperSampleSpacing = this->TumorGrowthNode->GetSuperSampled_Spacing();
    if (SuperSampleSpacing <= 0.0) {
      ROI->Delete();
      ROIExtent->Delete();
      return NULL;
    }
  }

  vtkImageResample *ROISuperSample = vtkImageResample::New(); 
     ROISuperSample->SetDimensionality(3);
     ROISuperSample->SetInterpolationModeToLinear();
     ROISuperSample->SetInput(ROIExtent->GetOutput());
     ROISuperSample->SetAxisOutputSpacing(0,SuperSampleSpacing/Spacing[0]);
     ROISuperSample->SetAxisOutputSpacing(1,SuperSampleSpacing/Spacing[1]);
     ROISuperSample->SetAxisOutputSpacing(2,SuperSampleSpacing/Spacing[2]);
     ROISuperSample->ReleaseDataFlagOff();
  ROISuperSample->Update();


  vtkImageChangeInformation *ROISuperSampleExtent = vtkImageChangeInformation::New();
     ROISuperSampleExtent->SetInput(ROISuperSample->GetOutput());
     ROISuperSampleExtent->SetOutputSpacing(1,1,1);
  ROISuperSampleExtent->Update();

  //  set TumorGrowth(scan${ID},save,Name) "$TumorGrowth(scan${ID},save,Name)_SuperSample" 
 
  // ---------------------------------
  // Now return results and clean up 
  char VolumeOutputName[1024];
  if (ScanNum > 1) sprintf(VolumeOutputName, "TG_scan2_Global_SuperSampled");
  else sprintf(VolumeOutputName, "TG_scan1_SuperSampled");

  vtkMRMLScalarVolumeNode *VolumeOutputNode = this->CreateVolumeNode(volumeNode,VolumeOutputName);
  // VolumeOutputNode->SetAndObserveImageData(ROIExtent->GetOutput());

  VolumeOutputNode->SetAndObserveImageData(ROISuperSampleExtent->GetOutput());
  VolumeOutputNode->SetSpacing(SuperSampleSpacing,SuperSampleSpacing,SuperSampleSpacing); 

  // Compute new rjk matrix 
  // double IJKToRASDirections[3][3];

  // Set new orgin
  vtkMatrix4x4 *ijkToRAS=vtkMatrix4x4::New();
  volumeNode->GetIJKToRASMatrix(ijkToRAS);
  double newIJKOrigin[4] = {this->TumorGrowthNode->GetROIMin(0),this->TumorGrowthNode->GetROIMin(1), this->TumorGrowthNode->GetROIMin(2), 1.0 };
  double newRASOrigin[4];
  ijkToRAS->MultiplyPoint(newIJKOrigin,newRASOrigin);
  VolumeOutputNode->SetOrigin(newRASOrigin[0],newRASOrigin[1],newRASOrigin[2]);


  // In tcl
  // set GUI  [$::slicer3::Application GetModuleGUIByName "TumorGrowth"]
  //    set NODE [$GUI  GetNode]
  //    set SCENE [$NODE GetScene]
  // set VolumeOutputNode [$SCENE GetNodeByID [$NODE GetScan1_SuperSampleRef]]
  // set VolumeOutputDisplayNode [$VolumeOutputNode GetScalarVolumeDisplayNode]
  // Important files  
  // ~/Slicer/Slicer3/Base/Logic/vtkSlicerVolumesLogic
  // ~/Slicer/Slicer3/Libs/MRML/vtkMRMLVolumeNode.h
  
  // this->TumorGrowthNode->GetScene()->AddNode(VolumeOutputNode);
  // VolumeOutputNode->Delete();

  ROISuperSampleExtent->Delete();
  ROISuperSample->Delete();
  ROIExtent->Delete();
  ROI->Delete();
  // VolumeOutputNode->PrintSelf(cout , indent);

  return VolumeOutputNode;
}

void vtkTumorGrowthLogic::SourceAnalyzeTclScripts(vtkSlicerApplication *app) {
 char TCL_FILE[1024]; 
 // Kilian: Can we copy this over to the build directory
 // cout - later, when it works do it this way bc more 
 sprintf(TCL_FILE,"%s/Modules/TumorGrowth/tcl/TumorGrowthFct.tcl",vtksys::SystemTools::GetEnv("SLICER_HOME"));
 // sprintf(TCL_FILE,"%s/../Slicer3/Modules/TumorGrowth/tcl/TumorGrowthFct.tcl",vtksys::SystemTools::GetEnv("SLICER_HOME"));

 app->LoadScript(TCL_FILE); 
 // later do it this way 
 sprintf(TCL_FILE,"%s/Modules/TumorGrowth/tcl/TumorGrowthReg.tcl",vtksys::SystemTools::GetEnv("SLICER_HOME"));
 // sprintf(TCL_FILE,"%s/../Slicer3/Modules/TumorGrowth/tcl/TumorGrowthReg.tcl",vtksys::SystemTools::GetEnv("SLICER_HOME"));
 app->LoadScript(TCL_FILE); 
}

void vtkTumorGrowthLogic::DeleteAnalyzeOutput(vtkSlicerApplication *app) {
   // Delete old attached node first 
  if (!TumorGrowthNode) return;
  this->SourceAnalyzeTclScripts(app);

  app->Script("::TumorGrowthTcl::Scan2ToScan1Registration_DeleteOutput Global");

  vtkMRMLVolumeNode* currentNode =  vtkMRMLVolumeNode::SafeDownCast(this->TumorGrowthNode->GetScene()->GetNodeByID(this->TumorGrowthNode->GetScan2_SuperSampleRef()));
  if (currentNode) { 
    this->TumorGrowthNode->GetScene()->RemoveNode(currentNode); 
    this->TumorGrowthNode->SetScan2_SuperSampleRef(NULL);
  }

  app->Script("::TumorGrowthTcl::HistogramNormalization_DeleteOutput"); 
  app->Script("::TumorGrowthTcl::Scan2ToScan1Registration_DeleteOutput Local"); 
  app->Script("::TumorGrowthTcl::IntensityThresholding_DeleteOutput 1");
  app->Script("::TumorGrowthTcl::IntensityThresholding_DeleteOutput 2");
  app->Script("::TumorGrowthTcl::Analysis_Intensity_DeleteOutput"); 
}

int vtkTumorGrowthLogic::AnalyzeGrowth(vtkSlicerApplication *app) {
  // This is for testing how to start a tcl script 
  cout << "=== Start ANALYSIS ===" << endl;

  // vtkIndent indent;
  // this->TumorGrowthNode->PrintSelf(cout,indent);
  // cout << " ======================" << endl;


  this->SourceAnalyzeTclScripts(app);
  
  if (0) { 
  cout << "=== 1 ===" << endl;
  app->Script("::TumorGrowthTcl::Scan2ToScan1Registration_GUI Global");


  //----------------------------------------------
  // Second step -> Save the outcome
  if (!this->TumorGrowthNode) {return 0;}
  

  {
     // Delete old attached node first 
     vtkMRMLVolumeNode* currentNode =  vtkMRMLVolumeNode::SafeDownCast(this->TumorGrowthNode->GetScene()->GetNodeByID(this->TumorGrowthNode->GetScan2_SuperSampleRef()));
     if (currentNode) { this->TumorGrowthNode->GetScene()->RemoveNode(currentNode); }
  }

  vtkMRMLScalarVolumeNode *outputNode = this->CreateSuperSample(2,app);
  if (!outputNode) {return 0;} 
  this->TumorGrowthNode->SetScan2_SuperSampleRef(outputNode->GetID());
  this->SaveVolume(app,outputNode);

  //----------------------------------------------

  cout << "=== 2 ===" << endl;
  app->Script("::TumorGrowthTcl::HistogramNormalization_GUI"); 
  cout << "=== 3 ===" << endl;
  app->Script("::TumorGrowthTcl::Scan2ToScan1Registration_GUI Local"); 
  } else {
    if (!this->TumorGrowthNode->GetScan2_LocalRef() || !strcmp(this->TumorGrowthNode->GetScan2_LocalRef(),"")) { 
      char fileName[1024];
      sprintf(fileName,"%s/TG_scan2_Local.nhdr",this->TumorGrowthNode->GetWorkingDir());
      vtkMRMLVolumeNode* tmp = this->LoadVolume(app,fileName,0,"TG_scan2_Local");
      if (tmp) {
    this->TumorGrowthNode->SetScan2_LocalRef(tmp->GetID());
      } else {
    cout << "Error: Could not load " << fileName << endl;
    return 0;
      }
    }
  }
  if (this->TumorGrowthNode->GetAnalysis_Intensity_Flag()) { 
    cout << "=== 4 ===" << endl;
    if (!atoi(app->Script("::TumorGrowthTcl::IntensityThresholding_GUI 1"))) return 0; 
    cout << "=== 5 ===" << endl;
    if (!atoi(app->Script("::TumorGrowthTcl::IntensityThresholding_GUI 2"))) return 0; 
    cout << "=== INTENSITY ANALYSIS ===" << endl;
    if (!atoi(app->Script("::TumorGrowthTcl::Analysis_Intensity_GUI"))) return 0; 
  } 
  if (this->TumorGrowthNode->GetAnalysis_Deformable_Flag()) {
    cout << "=== DEFORMABLE ANALYSIS ===" << endl;
    if (!atoi(app->Script("::TumorGrowthTcl::Analysis_Deformable_GUI"))) return 0; 
  }
  cout << "=== End ANALYSIS ===" << endl;

  return 1;
}

void vtkTumorGrowthLogic::RegisterMRMLNodesWithScene() {
   vtkMRMLTumorGrowthNode* tmNode =  vtkMRMLTumorGrowthNode::New();
   this->GetMRMLScene()->RegisterNodeClass(tmNode);
   tmNode->Delete();
}

vtkImageThreshold* vtkTumorGrowthLogic::CreateAnalysis_Intensity_Final() {
  if (this->Analysis_Intensity_Final) { this->Analysis_Intensity_Final->Delete(); }
  this->Analysis_Intensity_Final = vtkImageThreshold::New();
  return this->Analysis_Intensity_Final;
}

vtkImageThreshold* vtkTumorGrowthLogic::CreateAnalysis_Intensity_ROINegativeBin() {
  if (this->Analysis_Intensity_ROINegativeBin) { this->Analysis_Intensity_ROINegativeBin->Delete(); }
  this->Analysis_Intensity_ROINegativeBin = vtkImageThreshold::New();
  return this->Analysis_Intensity_ROINegativeBin;
}

vtkImageThreshold* vtkTumorGrowthLogic::CreateAnalysis_Intensity_ROIPositiveBin() {
  if (this->Analysis_Intensity_ROIPositiveBin) { this->Analysis_Intensity_ROIPositiveBin->Delete(); }
  this->Analysis_Intensity_ROIPositiveBin = vtkImageThreshold::New();
  return this->Analysis_Intensity_ROIPositiveBin;
}

vtkImageMathematics* vtkTumorGrowthLogic::CreateAnalysis_Intensity_ROIBinReal() {
  if (this->Analysis_Intensity_ROIBinReal) { this->Analysis_Intensity_ROIBinReal->Delete(); }
  this->Analysis_Intensity_ROIBinReal = vtkImageMathematics::New();
  return this->Analysis_Intensity_ROIBinReal;
}

vtkImageSumOverVoxels* vtkTumorGrowthLogic::CreateAnalysis_Intensity_ROITotal() {
  if (this->Analysis_Intensity_ROITotal) { this->Analysis_Intensity_ROITotal->Delete(); }
  this->Analysis_Intensity_ROITotal = vtkImageSumOverVoxels::New();
  return this->Analysis_Intensity_ROITotal;
}

double vtkTumorGrowthLogic::MeassureGrowth(vtkSlicerApplication *app) {
  
  if (!this->Analysis_Intensity_Final || !this->Analysis_Intensity_ROINegativeBin || !this->Analysis_Intensity_ROIPositiveBin || !this->Analysis_Intensity_ROITotal || !this->TumorGrowthNode ) return -1;
  app->Script("::TumorGrowthTcl::Analysis_Intensity_UpdateThreshold_GUI");
  // Just for display 
  this->Analysis_Intensity_Final->ThresholdByUpper(this->Analysis_Intensity_Threshold); 
  this->Analysis_Intensity_Final->Update();
  this->Analysis_Intensity_ROINegativeBin->ThresholdByLower(-this->Analysis_Intensity_Threshold); 
  this->Analysis_Intensity_ROINegativeBin->Update(); 
  this->Analysis_Intensity_ROIPositiveBin->ThresholdByUpper(this->Analysis_Intensity_Threshold); 
  this->Analysis_Intensity_ROIPositiveBin->Update(); 
  this->Analysis_Intensity_ROITotal->Update(); 
  return this->Analysis_Intensity_ROITotal->GetVoxelSum(); 
}

void vtkTumorGrowthLogic::SaveVolume(vtkSlicerApplication *app, vtkMRMLVolumeNode *volNode) {
  if (!this->SaveVolumeFlag) return;  
  this->SaveVolumeForce(app,volNode);
}


void vtkTumorGrowthLogic::SaveVolumeFileName( vtkMRMLVolumeNode *volNode, char *FileName) {
  sprintf(FileName,"%s/%s.nhdr",this->TumorGrowthNode->GetWorkingDir(),volNode->GetName());
}

void vtkTumorGrowthLogic::SaveVolumeForce(vtkSlicerApplication *app, vtkMRMLVolumeNode *volNode) {
 // Initialize
 vtkSlicerVolumesGUI  *volumesGUI    = vtkSlicerVolumesGUI::SafeDownCast(app->GetModuleGUIByName("Volumes")); 
 if (!volumesGUI) return;
 vtkSlicerVolumesLogic *volumesLogic = volumesGUI->GetLogic();

 // Create Directory if necessary
 {
   char CMD[1024];
   sprintf(CMD,"file isdirectory %s",this->TumorGrowthNode->GetWorkingDir()); 
   if (!atoi(app->Script(CMD))) { 
     sprintf(CMD,"file mkdir %s",this->TumorGrowthNode->GetWorkingDir()); 
     app->Script(CMD); 
   }
 }

 {
   char fileName[1024];
   this->SaveVolumeFileName(volNode,fileName);
   cout << "vtkTumorGrowthLogic::SaveVolume: Saving File :" << fileName << endl;
   if (!volumesLogic->SaveArchetypeVolume( fileName, volNode ) )  {
     cout << "Error: Could no save file " << endl;
   }
 }
}

vtkMRMLVolumeNode* vtkTumorGrowthLogic::LoadVolume(vtkSlicerApplication *app, char* fileName, int LabelMapFlag,char* volumeName) {
   vtkSlicerVolumesGUI  *volumesGUI    = vtkSlicerVolumesGUI::SafeDownCast(app->GetModuleGUIByName("Volumes")); 
   if (!volumesGUI) return NULL;
   vtkSlicerVolumesLogic *volumesLogic = volumesGUI->GetLogic();
   // Ignore error messages - I do not know how to get around them 
   return volumesLogic->AddArchetypeVolume(fileName,0,LabelMapFlag,volumeName);
}




//----------------------------------------------------------------------------
void vtkTumorGrowthLogic::PrintResult(ostream& os, vtkSlicerApplication *app)
{  
  // vtkMRMLNode::PrintSelf(os,indent);
  if (!this->TumorGrowthNode) return;
  os  << "This file was generated by vtkMrmTumorGrowthNode " << "\n";;
  os  << "Date:      " << app->Script("date") << "\n";;

  vtkMRMLVolumeNode *VolNode = vtkMRMLVolumeNode::SafeDownCast(this->TumorGrowthNode->GetScene()->GetNodeByID(this->TumorGrowthNode->GetScan1_Ref()));
  os  << "Scan1_Ref: " <<  (VolNode && VolNode->GetStorageNode() ? VolNode->GetStorageNode()->GetFileName() : "(none)") << "\n";

  VolNode = vtkMRMLVolumeNode::SafeDownCast(this->TumorGrowthNode->GetScene()->GetNodeByID(this->TumorGrowthNode->GetScan2_Ref()));
  os  << "Scan2_Ref: " <<  (VolNode && VolNode->GetStorageNode() ? VolNode->GetStorageNode()->GetFileName() : "(none)") << "\n";
  os  << "ROI:" << endl;
  os  << "  Min: " << this->TumorGrowthNode->GetROIMin(0) << " "<< this->TumorGrowthNode->GetROIMin(1) << " "<< this->TumorGrowthNode->GetROIMin(2) <<"\n";
  os  << "  Max: " << this->TumorGrowthNode->GetROIMax(0) << " "<< this->TumorGrowthNode->GetROIMax(1) << " "<< this->TumorGrowthNode->GetROIMax(2) <<"\n";
  os  << "Threshold: [" << this->TumorGrowthNode->GetSegmentThresholdMin() <<", " << this->TumorGrowthNode->GetSegmentThresholdMax() << "]\n";
  if (this->TumorGrowthNode->GetAnalysis_Intensity_Flag()) {
    os  << "Analysis based on Intensity Pattern" << endl;
    os  << "  Sensitivity:      "<< this->TumorGrowthNode->GetAnalysis_Intensity_Sensitivity() << "\n";
    double Growth = this->MeassureGrowth(app); 
    os  << "  Intensity Metric: "<<  floor(Growth*this->TumorGrowthNode->GetSuperSampled_VoxelVolume()*1000)/1000.0 << "mm" << char(179) 
       << " (" << int(Growth*this->TumorGrowthNode->GetSuperSampled_RatioNewOldSpacing()) << " Voxels)" << "\n";
  }
  if (this->TumorGrowthNode->GetAnalysis_Deformable_Flag()) {
    os  << "Analysis based on Deformable Map" << endl;
    os  << "  Segmentation Metric: "<<  floor(this->TumorGrowthNode->GetAnalysis_Deformable_SegmentationGrowth()*1000)/1000.0 << "mm" << char(179) 
       << " (" << int(this->TumorGrowthNode->GetAnalysis_Deformable_SegmentationGrowth()/this->TumorGrowthNode->GetScan1_VoxelVolume()) << " Voxels)\n";
    os  << "  Jacobian Metric:     "<<  floor(this->TumorGrowthNode->GetAnalysis_Deformable_JacobianGrowth()*1000)/1000.0 << "mm" << char(179) 
       << " (" << int(this->TumorGrowthNode->GetAnalysis_Deformable_JacobianGrowth()/this->TumorGrowthNode->GetScan1_VoxelVolume()) << " Voxels)\n";
  }
}

// works for running stuff in TCL so that you do not need to look in two windows 
void vtkTumorGrowthLogic::PrintText(char *TEXT) {
  cout << TEXT << endl;
} 
  


