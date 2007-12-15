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

//#include "vtkSlicerVolumesLogic.h"
//#include "vtkSlicerVolumesGUI.h"
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

}

//----------------------------------------------------------------------------
vtkTumorGrowthLogic::~vtkTumorGrowthLogic()
{
  vtkSetMRMLNodeMacro(this->TumorGrowthNode, NULL);
  this->SetProgressCurrentAction(NULL);
  this->SetModuleName(NULL);
}

//----------------------------------------------------------------------------
void vtkTumorGrowthLogic::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
  // !!! todo
}


//----------------------------------------------------------------------------
void
vtkTumorGrowthLogic::
StartAnalysis()
{
  //
  // make sure preprocessing is up to date
  //
  std::cerr << "Start preprocessing..." << std::endl;
  // this->StartPreprocessing();
  std::cerr << "Done preprocessing." << std::endl;

  //
  // make sure we're ready to start
  //

  // find output volume
  if (!this->TumorGrowthNode) 
    {
    vtkErrorMacro("Segmenter node is null---aborting segmentation.");
    return;
    }
//   vtkMRMLScalarVolumeNode *outVolume =  this->TumorGrowthNode->GetOutputVolumeNode();
//   if (outVolume == NULL)
//     {
//     vtkErrorMacro("No output volume found---aborting segmentation.");
//     return;
//     }
// 
//   //
//   // Copy RASToIJK matrix, and other attributes from input to
//   // output. Use first target volume as source for this data.
//   //
//   
//   // get attributes from first target input volume
//   const char* inMRLMID = 
//     this->TumorGrowthNode->GetTargetNode()->GetNthVolumeNodeID(0);
//   vtkMRMLScalarVolumeNode *inVolume = vtkMRMLScalarVolumeNode::
//     SafeDownCast(this->GetMRMLScene()->GetNodeByID(inMRLMID));
//   if (inVolume == NULL)
//     {
//     vtkErrorMacro("Can't get first target image.");
//     return;
//     }
// 
//   outVolume->CopyOrientation(inVolume);
//   outVolume->SetAndObserveTransformNodeID(inVolume->GetTransformNodeID());
//   outVolume->SetName("Segmentation_Result");


  // PERFORM ANALYSIS

  // Copy Results to output volume 
  //vtkImageData* image = vtkImageData::New(); 
  //image->DeepCopy(segmenter->GetOutput());
  //outVolume->SetAndObserveImageData(image);
  //image->Delete();
  //outVolume->SetModifiedSinceRead(1);

  // Originally had this here 
  //outVolume-
  //SetImageData(segmenter->GetOutput());

  //
  // save template file if desired
  //
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

  char *VolumeRef = (ScanNum > 1 ? this->TumorGrowthNode->GetScan2_Ref() : this->TumorGrowthNode->GetScan1_Ref()); 
  vtkMRMLVolumeNode* volumeNode = vtkMRMLVolumeNode::SafeDownCast(this->TumorGrowthNode->GetScene()->GetNodeByID(VolumeRef));
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
  if (ScanNum == 1) {
    double *Spacing = volumeNode->GetImageData()->GetSpacing(); 
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
     ROISuperSample->SetAxisOutputSpacing(0,SuperSampleSpacing);
     ROISuperSample->SetAxisOutputSpacing(1,SuperSampleSpacing);
     ROISuperSample->SetAxisOutputSpacing(2,SuperSampleSpacing);
     ROISuperSample->ReleaseDataFlagOff();
  ROISuperSample->Update();
       
  //  set TumorGrowth(scan${ID},save,Name) "$TumorGrowth(scan${ID},save,Name)_SuperSample" 
 
  // ---------------------------------
  // Now return results and clean up 
  char VolumeOutputName[1024];
  if (ScanNum > 1) sprintf(VolumeOutputName, "TG_scan2_SuperSampled");
  else sprintf(VolumeOutputName, "TG_scan1_SuperSampled");

  vtkMRMLScalarVolumeNode *VolumeOutputNode = vtkMRMLScalarVolumeNode::New();
  VolumeOutputNode->SetName(VolumeOutputName);
  VolumeOutputNode->SetAndObserveImageData(ROISuperSample->GetOutput());
  this->TumorGrowthNode->GetScene()->AddNode(VolumeOutputNode);
  // KILIAN: image is not correctly displayed later - what should we do ? 

  VolumeOutputNode->Delete();
  ROISuperSample->Delete();
  ROIExtent->Delete();
  ROI->Delete();
  // VolumeOutputNode->PrintSelf(cout , indent);

  return VolumeOutputNode;
}


void vtkTumorGrowthLogic::AnalyzeGrowth(vtkSlicerApplication *app) {

  // This is for testing how to start a tcl script 
  cout << "=== Start ANALYSIS ===" << endl;
  char TCL_FILE[1024]; 
  sprintf(TCL_FILE,"%s/../Slicer3/Modules/TumorGrowth/tcl/TumorGrowthFct.tcl",vtksys::SystemTools::GetEnv("SLICER_HOME"));
  app->LoadScript(TCL_FILE); 
  app->Script("::TumorGrowthTcl::Scan2ToScan1Registration_GUI Global");
  this->CreateSuperSample(2,app);
  app->Script("::TumorGrowthTcl::HistogramNormalization_GUI"); 
  app->Script("::TumorGrowthTcl::Scan2ToScan1Registration_GUI Local"); 
  app->Script("::TumorGrowthTcl::IntensityThresholding_GUI 1"); 
  app->Script("::TumorGrowthTcl::IntensityThresholding_GUI 2"); 

  // set SCAN1 [TumorGrowth(Scan1,ROIThreshold) GetOutput]
  // set SCAN2 [TumorGrowth(Scan2,ROIThreshold) GetOutput]
  // set SEGMENT [TumorGrowth(Scan1,Segment) GetOutput]
  app->Script("::TumorGrowthTcl::AnalysisIntensity $SCAN1 $SEGMENT $SCAN2 2"); 
  cout << "=== End ANALYSIS ===" << endl;
}


//-----------------------------------------------------------------------------
void
vtkTumorGrowthLogic::
PopulateTestingData()
{
  vtkDebugMacro("Begin populating test data");

  //
  // add some nodes to the hierarchy
  //
  vtkDebugMacro("Setting parameters for root node");
  // vtkIdType rootNodeID         = this->MRMLManager->GetTreeRootNodeID();
  //this->MRMLManager->SetTreeNodeLabel(rootNodeID, "Root");
  //this->MRMLManager->SetTreeNodeName(rootNodeID, "Root");

  // vtkDebugMacro("Setting save parameters");
  // this->MRMLManager->SetSaveWorkingDirectory("/tmp");
  // this->MRMLManager->SetSaveTemplateFilename("/tmp/TumorGrowthTemplate.mrml");
  // this->MRMLManager->SetSaveTemplateAfterSegmentation(1);
  // this->MRMLManager->SetSaveIntermediateResults(1);
  // this->MRMLManager->SetSaveSurfaceModels(1);
  
  // this->SetProgressGlobalFractionCompleted(0.9);

  vtkDebugMacro("Done populating test data");
}

void vtkTumorGrowthLogic::RegisterMRMLNodesWithScene() {
   vtkMRMLTumorGrowthNode* tmNode =  vtkMRMLTumorGrowthNode::New();
   this->GetMRMLScene()->RegisterNodeClass(tmNode);
   tmNode->Delete();
}


// according to vtkGradnientAnisotrpoicDiffusionoFilterGUI
// 
//void vtkTumorGrowthLogic::ProcessMRMLEvents(vtkObject* caller, unsigned long event, void* callData) {
//  vtkMRMLTumorGrowthNode* node = vtkMRMLTumorGrowthNode::SafeDownCast(caller);
//  cout << " vtkTumorGrowthLogic::ProcessMRMLEvents " << endl;
  //if (node != NULL && this->GetTumorGrowthNode() == node) 
  //  {
  //  this->UpdateGUI();
  //  }
// }


// This function is automatically called as part of the pipeline 
// According to EMSegment
// void vtkTumorGrowthLogic::ProcessMRMLEvents(vtkObject* caller, unsigned long event, void* callData) {
//   if (vtkMRMLScene::SafeDownCast(caller) != this->MRMLScene) {
//     return;
//   }
// 
//   vtkMRMLNode *node = (vtkMRMLNode*)(callData);
//   if (node == NULL)
//     {
//     return;
//     }
// 
//   if (event == vtkMRMLScene::NodeAddedEvent)
//   {
//       if (node->IsA("vtkMRMLTumorGrowthNode"))  
//       {
//      // update node in GUI   
//      this->NODE = (vtkMRMLTumorGrowthNode*) node;
//      Update 
// 
//   ///   
//   look at vtkGradnientAnisotrpoicDiffusionoFilterGUI
// 
//   vtkSetAndObserveMRMLNodeMacro( this->GradientAnisotropicDiffusionFilterNode, n);
//      (vtkMRMLTumorGrowthNode* node);
//
//         cout << "Update Now" << endl;
//       
//       }
//   }
//   else if (event == vtkMRMLScene::NodeRemovedEvent)
//   {
//     if (node->IsA("vtkMRMLTumorGrowthNode"))  
//       {
//        cout << "Remove Now" << endl;
//       }
//   }
// }



