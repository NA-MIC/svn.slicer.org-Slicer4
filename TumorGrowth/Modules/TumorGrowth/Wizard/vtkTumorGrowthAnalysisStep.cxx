#include "vtkTumorGrowthAnalysisStep.h"

#include "vtkTumorGrowthGUI.h"
#include "vtkMRMLTumorGrowthNode.h"

#include "vtkKWWizardWidget.h"
#include "vtkKWWizardWorkflow.h"
#include "vtkKWThumbWheel.h"

#include "vtkKWFrameWithLabel.h"
#include "vtkKWLabel.h"
#include "vtkKWEntry.h"
//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkTumorGrowthAnalysisStep);
vtkCxxRevisionMacro(vtkTumorGrowthAnalysisStep, "$Revision: 1.2 $");

//----------------------------------------------------------------------------
vtkTumorGrowthAnalysisStep::vtkTumorGrowthAnalysisStep()
{
  this->SetName("Analysis"); 
  this->SetDescription("Analysis of Tumor Growth"); 

  this->SensitivityScale = NULL;
}

//----------------------------------------------------------------------------
vtkTumorGrowthAnalysisStep::~vtkTumorGrowthAnalysisStep()
{
  if (this->SensitivityScale)
    {
    this->SensitivityScale->Delete();
    this->SensitivityScale = NULL;
    }
}

//----------------------------------------------------------------------------
void vtkTumorGrowthAnalysisStep::ShowUserInterface()
{
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
    this->SensitivityScale->SetRange(0.0, 60.0);
    this->SensitivityScale->SetMinimumValue(0.0);
    this->SensitivityScale->ClampMinimumValueOn(); 
    this->SensitivityScale->SetMaximumValue(60.0);
    this->SensitivityScale->ClampMaximumValueOn(); 
    this->SensitivityScale->SetResolution(50);
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

  this->Script( "pack %s -side top -anchor nw -padx 2 -pady 2", this->SensitivityScale->GetWidgetName());
  // this->Script("grid %s -column 0 -row 0 -sticky nw -padx 2 -pady 2", this->SensitivityScale->GetWidgetName());

  {
    vtkKWWizardWidget *wizard_widget = this->GetGUI()->GetWizardWidget();
    // wizard_widget->GetOKButton()->SetText("Run");
    wizard_widget->GetCancelButton()->SetText("OK"); 
    wizard_widget->GetCancelButton()->SetCommand(this, "ResetPipelineCallback");
    wizard_widget->GetCancelButton()->EnabledOn();
    wizard_widget->OKButtonVisibilityOff();

  }

}


//----------------------------------------------------------------------------
void vtkTumorGrowthAnalysisStep::SensitivityChangedCallback(vtkIdType sel_vol_id, double value)
{
  // Sensitivity has changed because of user interaction 
  vtkMRMLTumorGrowthNode *mrmlNode = this->GetGUI()->GetNode();
  cout << "vtkTumorGrowthAnalysisStep::SensitivityChangedCallback ---- Please set correctly" << endl;
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
