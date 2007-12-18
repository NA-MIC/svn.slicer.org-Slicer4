#include "vtkTumorGrowthSegmentationStep.h"

#include "vtkTumorGrowthGUI.h"
#include "vtkMRMLTumorGrowthNode.h"

#include "vtkKWWizardWidget.h"
#include "vtkKWWizardWorkflow.h"
#include "vtkKWThumbWheel.h"
#include "vtkKWFrameWithLabel.h"
#include "vtkKWLabel.h"
#include "vtkKWEntry.h"
#include "vtkTumorGrowthLogic.h"
#include "vtkSlicerApplicationGUI.h"
#include "vtkSlicerSliceControllerWidget.h"
#include "vtkKWScale.h"
#include "vtkImageAccumulate.h"
#include "vtkImageThreshold.h"
#include "vtkSlicerVolumesLogic.h" 
#include "vtkSlicerVolumesGUI.h"
#include "vtkSlicerApplication.h"
#include "vtkImageIslandFilter.h"
//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkTumorGrowthSegmentationStep);
vtkCxxRevisionMacro(vtkTumorGrowthSegmentationStep, "$Revision: 1.2 $");

//----------------------------------------------------------------------------
vtkTumorGrowthSegmentationStep::vtkTumorGrowthSegmentationStep()
{
  this->SetName("3/4. Identify Tumor in First Scan"); 
  this->SetDescription("Move slider to outline boundary of tumor."); 

  this->ThresholdScale = NULL;

  this->PreSegment = NULL;
  this->PreSegmentNode = NULL;
  this->SegmentNode = NULL;


}

//----------------------------------------------------------------------------
vtkTumorGrowthSegmentationStep::~vtkTumorGrowthSegmentationStep()
{
  if (this->ThresholdScale)
    {
    this->ThresholdScale->Delete();
    this->ThresholdScale = NULL;
    }
  this->PreSegmentScan1Remove();
  this->SegmentScan1Remove();
}

//----------------------------------------------------------------------------
void vtkTumorGrowthSegmentationStep::ShowUserInterface()
{
  // ----------------------------------------
  // Display Super Sampled Volume 
  // ----------------------------------------
  
  vtkMRMLTumorGrowthNode* node = this->GetGUI()->GetNode();
  int intMin, intMax;

  if (node) { 
    vtkMRMLVolumeNode *volumeNode = vtkMRMLVolumeNode::SafeDownCast(node->GetScene()->GetNodeByID(node->GetScan1_SuperSampleRef()));
    if (volumeNode) {
      vtkSlicerApplicationLogic *applicationLogic = this->GetGUI()->GetLogic()->GetApplicationLogic();
      applicationLogic->GetSelectionNode()->SetActiveVolumeID(volumeNode->GetID());
      applicationLogic->PropagateVolumeSelection();

      vtkImageAccumulate *blub = vtkImageAccumulate::New();
      blub->SetInput(volumeNode->GetImageData());
      blub->Update();
      intMin = int(blub->GetMin()[0]);
      intMax = int(blub->GetMax()[0]);
      blub->Delete();
    } else {
      intMin = 0;
      intMax = 0;      
    } 
  } else {
      intMin = 0;
      intMax = 0;      
  }

  // ----------------------------------------
  // Build GUI 
  // ----------------------------------------
  this->vtkTumorGrowthStep::ShowUserInterface();
  this->Frame->SetLabelText("Identify Tumor");
  this->Script("pack %s -side top -anchor nw -fill x -padx 0 -pady 2", this->Frame->GetWidgetName());

  if (!this->ThresholdScale)
    {
    this->ThresholdScale = vtkKWThumbWheel::New();
    }
  if (!this->ThresholdScale->IsCreated())
  {
    this->ThresholdScale->SetParent(this->Frame->GetFrame());
    this->ThresholdScale->Create();
    this->ThresholdScale->ClampMinimumValueOn(); 
    this->ThresholdScale->ClampMaximumValueOn(); 
    this->ThresholdScale->SetResolution(50);
    this->ThresholdScale->SetLinearThreshold(1);
    this->ThresholdScale->SetThumbWheelSize(TUMORGROWTH_WIDGETS_SLIDER_WIDTH,TUMORGROWTH_WIDGETS_SLIDER_HEIGHT);
    this->ThresholdScale->DisplayEntryOn();
    this->ThresholdScale->DisplayLabelOn();
    this->ThresholdScale->GetLabel()->SetText("Threshold");
    this->ThresholdScale->SetCommand(this,"ThresholdChangedCallback");
    this->ThresholdScale->DisplayEntryAndLabelOnTopOff(); 
    this->ThresholdScale->SetBalloonHelpString("Move wheel to segment tumor");
    // KLIIAN: Read from MRML File 
  }

  this->ThresholdScale->SetRange(intMin, intMax);
  this->ThresholdScale->SetMinimumValue(intMin);
  this->ThresholdScale->SetMaximumValue(intMax);

  vtkMRMLTumorGrowthNode *mrmlNode = this->GetGUI()->GetNode();
  if (mrmlNode && (mrmlNode->GetSegmentThreshold() > -1)) {
    this->ThresholdScale->SetValue(mrmlNode->GetSegmentThreshold());
  } else {
    this->ThresholdScale->SetValue((intMax - intMin)/2.0);
  }

  this->Script( "pack %s -side top -anchor nw -padx 2 -pady 2", this->ThresholdScale->GetWidgetName());

  // ----------------------------------------
  // Show segmentation 
  // ----------------------------------------
  this->PreSegmentScan1Define();
  // Necesary in order to transfere results from above lines  
  this->ThresholdChangedCallback(0);

  // this->TransitionCallback();   
}

void vtkTumorGrowthSegmentationStep::PreSegmentScan1Remove() {
  this->GetGUI()->GetMRMLScene()->RemoveNode(this->PreSegmentNode);
  this->PreSegmentNode = NULL;
  if (this->PreSegment) this->PreSegment->Delete();
  this->PreSegment = NULL;
}

void vtkTumorGrowthSegmentationStep::PreSegmentScan1Define() {
  // Initialize

  vtkMRMLTumorGrowthNode* Node      =  this->GetGUI()->GetNode();
  if (!Node) return;
  vtkMRMLVolumeNode *volumeNode = vtkMRMLVolumeNode::SafeDownCast(Node->GetScene()->GetNodeByID(Node->GetScan1_SuperSampleRef()));
  if (!volumeNode) return;
  if (!this->ThresholdScale) return;

  // Define LabelMap 
  if (this->PreSegment || this->PreSegmentNode) this->PreSegmentScan1Remove();
  this->PreSegment = vtkImageThreshold::New(); 
  this->PreSegment->SetInValue(10);
  this->PreSegment->SetOutValue(0);
  this->PreSegment->SetOutputScalarTypeToShort();
  this->PreSegment->SetInput(volumeNode->GetImageData());
  this->PreSegment->ThresholdByUpper(this->ThresholdScale->GetValue()); 
  this->PreSegment->Update();

  // show segmentation in Slicer 3 
  vtkSlicerApplicationGUI *applicationGUI     = this->GetGUI()->GetApplicationGUI();
  vtkSlicerApplicationLogic *applicationLogic = this->GetGUI()->GetLogic()->GetApplicationLogic();
  vtkSlicerVolumesLogic *volumesLogic         = (vtkSlicerVolumesGUI::SafeDownCast(vtkSlicerApplication::SafeDownCast(this->GetApplication())->GetModuleGUIByName("Volumes")))->GetLogic();

  if (this->PreSegmentNode) this->PreSegmentScan1Remove();
  this->PreSegmentNode = volumesLogic->CreateLabelVolume(Node->GetScene(),volumeNode, "TG_Scan1_PreSegment");
  this->PreSegmentNode->SetAndObserveImageData(this->PreSegment->GetOutput());

  // Later put in labelmap 
  applicationGUI->GetMainSliceGUI0()->GetSliceController()->GetForegroundSelector()->SetSelected(this->PreSegmentNode);
  applicationGUI->GetMainSliceGUI1()->GetSliceController()->GetForegroundSelector()->SetSelected(this->PreSegmentNode);
  applicationGUI->GetMainSliceGUI2()->GetSliceController()->GetForegroundSelector()->SetSelected(this->PreSegmentNode);
  applicationGUI->GetSlicesControlGUI()->GetSliceFadeScale()->SetValue(0.6);
  applicationLogic->PropagateVolumeSelection();

  this->SegmentScan1Remove();

  return;
}

void vtkTumorGrowthSegmentationStep::SegmentScan1Remove() {
  vtkMRMLTumorGrowthNode* Node = this->GetGUI()->GetNode();
  if (Node) {
    vtkMRMLVolumeNode* currentNode =  vtkMRMLVolumeNode::SafeDownCast(Node->GetScene()->GetNodeByID(Node->GetScan1_SegmentRef()));
    if (currentNode) this->GetGUI()->GetMRMLScene()->RemoveNode(currentNode); 
    Node->SetScan1_SegmentRef(NULL);
  }
  if (this->SegmentNode) {
    this->SegmentNode->Delete();
    this->SegmentNode = NULL;
  }
}

int vtkTumorGrowthSegmentationStep::SegmentScan1Define() {
  // Initialize
  if (!this->PreSegment || !this->PreSegmentNode) return 0;
  vtkMRMLTumorGrowthNode* Node = this->GetGUI()->GetNode();
  if (!Node) return 0 ;

  this->SegmentScan1Remove();

  vtkImageIslandFilter *RemoveIslands = vtkImageIslandFilter::New();
  RemoveIslands->SetIslandMinSize(1000);
    RemoveIslands->SetInput(this->PreSegment->GetOutput());
    RemoveIslands->SetNeighborhoodDim3D();
  RemoveIslands->Update(); 

  // Set It up 
  vtkSlicerVolumesLogic *volumesLogic         = (vtkSlicerVolumesGUI::SafeDownCast(vtkSlicerApplication::SafeDownCast(this->GetApplication())->GetModuleGUIByName("Volumes")))->GetLogic();

  this->SegmentNode = volumesLogic->CreateLabelVolume(Node->GetScene(),this->PreSegmentNode, "TG_scan1_Segment");
  this->SegmentNode->SetAndObserveImageData(RemoveIslands->GetOutput());

  RemoveIslands->Delete(); 
  this->PreSegmentScan1Remove();

  // Added it to MRML Script
  Node->SetScan1_SegmentRef(this->SegmentNode->GetID());

  return 1;
}


//----------------------------------------------------------------------------
void vtkTumorGrowthSegmentationStep::ThresholdChangedCallback(double value)
{
  if (!this->ThresholdScale || !this->PreSegment) return;
  PreSegment->ThresholdByUpper(this->ThresholdScale->GetValue()); 
  PreSegment->Update();
  // PreSegment->GetOutput()->Modified();
  this->PreSegmentNode->Modified();


  vtkMRMLTumorGrowthNode *mrmlNode = this->GetGUI()->GetNode();
  if (!mrmlNode) return;
  mrmlNode->SetSegmentThreshold(this->ThresholdScale->GetValue());

  // set GUI  [$::slicer3::Application GetModuleGUIByName "TumorGrowth"]
  // set STEP [$GUI GetSegmentationStep]
  // set FILT [$STEP GetPreSegment]

  // You can also watch MRML by doing 
  // MRMLWatcher m
  // parray MRML
  // $MRML(TG_scan1_SuperSampled) Print

}

//----------------------------------------------------------------------------
void vtkTumorGrowthSegmentationStep::TransitionCallback() 
{ 
  this->SegmentScan1Remove();
  if (!this->SegmentScan1Define()) return; 
  // Proceed to next step 
  this->GUI->GetWizardWidget()->GetWizardWorkflow()->AttemptToGoToNextStep();
}


//----------------------------------------------------------------------------
void vtkTumorGrowthSegmentationStep::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
