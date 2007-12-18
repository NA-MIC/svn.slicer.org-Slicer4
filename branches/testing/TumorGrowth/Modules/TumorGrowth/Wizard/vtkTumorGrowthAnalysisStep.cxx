#include "vtkTumorGrowthAnalysisStep.h"

#include "vtkTumorGrowthGUI.h"
#include "vtkMRMLTumorGrowthNode.h"

#include "vtkKWWizardWidget.h"
#include "vtkKWWizardWorkflow.h"
#include "vtkKWThumbWheel.h"

#include "vtkKWFrameWithLabel.h"
#include "vtkKWLabel.h"
#include "vtkKWEntry.h"
#include "vtkSlicerApplicationLogic.h"
#include "vtkTumorGrowthLogic.h"
#include "vtkSlicerSliceControllerWidget.h"
#include "vtkKWScale.h"
#include "vtkSlicerApplication.h"
//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkTumorGrowthAnalysisStep);
vtkCxxRevisionMacro(vtkTumorGrowthAnalysisStep, "$Revision: 1.2 $");

//----------------------------------------------------------------------------
vtkTumorGrowthAnalysisStep::vtkTumorGrowthAnalysisStep()
{
  this->SetName("Analysis"); 
  this->SetDescription("Analysis of Tumor Growth"); 

  this->SensitivityScale = NULL;
  this->GrowthLabel = NULL;

}

//----------------------------------------------------------------------------
vtkTumorGrowthAnalysisStep::~vtkTumorGrowthAnalysisStep()
{
  if (this->SensitivityScale)
    {
    this->SensitivityScale->Delete();
    this->SensitivityScale = NULL;
    }
  if (this->GrowthLabel) 
    {
      this->GrowthLabel->Delete();
      this->GrowthLabel = NULL;
    }


}

//----------------------------------------------------------------------------
void vtkTumorGrowthAnalysisStep::ShowUserInterface()
{
  // ----------------------------------------
  // Display Analysis Volume 
  // ----------------------------------------  
  vtkMRMLTumorGrowthNode* node = this->GetGUI()->GetNode();
  if (node) { 
    vtkMRMLVolumeNode *volumeSampleNode = vtkMRMLVolumeNode::SafeDownCast(node->GetScene()->GetNodeByID(node->GetScan1_SuperSampleRef()));
    vtkMRMLVolumeNode *volumeAnalysisNode = vtkMRMLVolumeNode::SafeDownCast(node->GetScene()->GetNodeByID(node->GetAnalysis_Ref()));
    if (volumeSampleNode && volumeAnalysisNode) {
      vtkSlicerApplicationLogic *applicationLogic = this->GetGUI()->GetLogic()->GetApplicationLogic();
      applicationLogic->GetSelectionNode()->SetActiveVolumeID(volumeSampleNode->GetID());

      vtkSlicerApplicationGUI *applicationGUI     = this->GetGUI()->GetApplicationGUI();
      applicationGUI->GetMainSliceGUI0()->GetSliceController()->GetForegroundSelector()->SetSelected(volumeAnalysisNode);
      applicationGUI->GetMainSliceGUI1()->GetSliceController()->GetForegroundSelector()->SetSelected(volumeAnalysisNode);
      applicationGUI->GetMainSliceGUI2()->GetSliceController()->GetForegroundSelector()->SetSelected(volumeAnalysisNode);
      applicationGUI->GetSlicesControlGUI()->GetSliceFadeScale()->SetValue(0.6);

      applicationLogic->PropagateVolumeSelection();
    } 
  }

  // ----------------------------------------
  // Build GUI 
  // ----------------------------------------

  this->vtkTumorGrowthStep::ShowUserInterface();

  this->Frame->SetLabelText("Tumor Growth");
  this->Script("pack %s -side top -anchor nw -fill x -padx 0 -pady 2", this->Frame->GetWidgetName());

  if (!this->SensitivityScale)
    {
    this->SensitivityScale = vtkKWThumbWheel::New();
    }
  if (!this->SensitivityScale->IsCreated())
  {
    this->SensitivityScale->SetParent(this->Frame->GetFrame());
    this->SensitivityScale->Create();
    this->SensitivityScale->SetRange(0.0,1.0);
    this->SensitivityScale->SetMinimumValue(0.0);
    this->SensitivityScale->ClampMinimumValueOn(); 
    this->SensitivityScale->SetMaximumValue(1.0);
    this->SensitivityScale->ClampMaximumValueOn(); 
    this->SensitivityScale->SetResolution(0.75);
    this->SensitivityScale->SetLinearThreshold(1);
    this->SensitivityScale->SetThumbWheelSize (TUMORGROWTH_WIDGETS_SLIDER_WIDTH,TUMORGROWTH_WIDGETS_SLIDER_HEIGHT);
    this->SensitivityScale->DisplayEntryOn();
    this->SensitivityScale->DisplayLabelOn();
    this->SensitivityScale->GetLabel()->SetText("Sensitivity");
    this->SensitivityScale->SetCommand(this,"SensitivityChangedCallback");
    this->SensitivityScale->DisplayEntryAndLabelOnTopOff(); 
    this->SensitivityScale->SetBalloonHelpString("The further the wheel is turned to the right the more robust the result");

    // this->SensitivityScale->GetEntry()->SetCommandTriggerToAnyChange();
  }

  // Initial value 
  vtkMRMLTumorGrowthNode *mrmlNode = this->GetGUI()->GetNode();
  if (mrmlNode) this->SensitivityScale->SetValue(mrmlNode->GetAnalysis_Sensitivity());

  this->Script( "pack %s -side top -anchor nw -padx 2 -pady 2", this->SensitivityScale->GetWidgetName());


  if (!this->GrowthLabel)
    {
    this->GrowthLabel = vtkKWLabel::New();
    }
  if (!this->GrowthLabel->IsCreated())
  {
    this->GrowthLabel->SetParent(this->Frame->GetFrame());
    this->GrowthLabel->Create();
  }
  this->Script( "pack %s -side top -anchor nw -padx 2 -pady 2", this->GrowthLabel->GetWidgetName());


  {
    vtkKWWizardWidget *wizard_widget = this->GetGUI()->GetWizardWidget();
    // wizard_widget->GetOKButton()->SetText("Run");
    wizard_widget->GetCancelButton()->SetText("OK"); 
    wizard_widget->GetCancelButton()->SetCommand(this, "ResetPipelineCallback");
    wizard_widget->GetCancelButton()->EnabledOn();
    wizard_widget->OKButtonVisibilityOff();

  }

  // Show results 
  this->SensitivityChangedCallback(0.0);
}


//----------------------------------------------------------------------------
void vtkTumorGrowthAnalysisStep::SensitivityChangedCallback(double value)
{
  // Sensitivity has changed because of user interaction
  vtkMRMLTumorGrowthNode *mrmlNode = this->GetGUI()->GetNode();
  if (!this->SensitivityScale || !mrmlNode || !this->GrowthLabel ) return;
  mrmlNode->SetAnalysis_Sensitivity(this->SensitivityScale->GetValue());
  double Growth = this->GetGUI()->GetLogic()->MeassureGrowth(vtkSlicerApplication::SafeDownCast(this->GetGUI()->GetApplication()));
  // show here 
  char TEXT[1024];
  // cout << "---------- " << Growth << " " << mrmlNode->GetSuperSampled_VoxelVolume() << " " << mrmlNode->GetSuperSampled_RatioNewOldSpacing() << endl;;
  sprintf(TEXT,"Growth: %.3f mm^3 (%d Voxels)", Growth*mrmlNode->GetSuperSampled_VoxelVolume(),int(Growth*mrmlNode->GetSuperSampled_RatioNewOldSpacing()));

  this->GrowthLabel->SetText(TEXT);
  // Show updated results 
  vtkMRMLVolumeNode *analysisNode = vtkMRMLVolumeNode::SafeDownCast(mrmlNode->GetScene()->GetNodeByID(mrmlNode->GetAnalysis_Ref()));
  if (analysisNode) analysisNode->Modified();
}

//----------------------------------------------------------------------------
void vtkTumorGrowthAnalysisStep::ResetPipelineCallback() 
{
  // Sensitivity has changed because of user interaction 
  vtkKWWizardWidget *wizard_widget = this->GetGUI()->GetWizardWidget();
  vtkKWWizardWorkflow *wizard_workflow = wizard_widget->GetWizardWorkflow();
  // Go Back to the beginning - you can also make this more generale by first getting the number of states 
  // and then doing a loop 
  wizard_workflow->AttemptToGoToPreviousStep();
  wizard_workflow->AttemptToGoToPreviousStep();
  wizard_workflow->AttemptToGoToPreviousStep();
  wizard_workflow->AttemptToGoToPreviousStep();
}

//----------------------------------------------------------------------------
void vtkTumorGrowthAnalysisStep::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
