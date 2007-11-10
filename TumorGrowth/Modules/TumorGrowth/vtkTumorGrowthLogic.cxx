#include <string>
#include <iostream>
#include <sstream>

#include "vtkObjectFactory.h"

#include "vtkTumorGrowthLogic.h"
#include "vtkTumorGrowth.h"

#include "vtkMRMLScene.h"

#include "vtkMRMLTumorGrowthNode.h"
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
  //outVolume->
  //SetImageData(segmenter->GetOutput());

  //
  // save template file if desired
  //
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



