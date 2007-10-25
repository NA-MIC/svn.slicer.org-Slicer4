#include "vtkTumorGrowthSegmentationStep.h"

#include "vtkTumorGrowthGUI.h"
#include "vtkMRMLTumorGrowthNode.h"

#include "vtkKWWizardWidget.h"
#include "vtkKWWizardWorkflow.h"
#include "vtkKWThumbWheel.h"
#include "vtkKWFrameWithLabel.h"
#include "vtkKWLabel.h"
#include "vtkKWEntry.h"

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkTumorGrowthSegmentationStep);
vtkCxxRevisionMacro(vtkTumorGrowthSegmentationStep, "$Revision: 1.2 $");

//----------------------------------------------------------------------------
vtkTumorGrowthSegmentationStep::vtkTumorGrowthSegmentationStep()
{
  this->SetName("3/4. Identify Tumor in First Scan"); 
  this->SetDescription("Move slider to outline boundary of tumor."); 

  this->ThresholdScale = NULL;
}

//----------------------------------------------------------------------------
vtkTumorGrowthSegmentationStep::~vtkTumorGrowthSegmentationStep()
{
  if (this->ThresholdScale)
    {
    this->ThresholdScale->Delete();
    this->ThresholdScale = NULL;
    }
}

//----------------------------------------------------------------------------
void vtkTumorGrowthSegmentationStep::ShowUserInterface()
{
  this->vtkTumorGrowthStep::ShowUserInterface();
  this->Frame->SetLabelText("Identify Tumor");
  this->Script("pack %s -side top -anchor nw -fill x -padx 0 -pady 2", this->Frame->GetWidgetName());

  if (!this->ThresholdScale)
    {
    this->ThresholdScale = vtkKWThumbWheel::New();
    }
  if (!this->ThresholdScale->IsCreated())
  {
    // this->ThresholdScale->SetParent(this->Frame->GetFrame());
    // this->ThresholdScale->PopupModeOn();
    // this->ThresholdScale->Create();
    // this->ThresholdScale->SetEntryWidth(4);
    // this->ThresholdScale->SetLabelText("Segment Tumor:");
    // this->ThresholdScale->GetLabel()->SetWidth(TUMORGROWTH_WIDGETS_LABEL_WIDTH - 9);
    // this->ThresholdScale->SetRange(0.0, 300.0);
    // this->ThresholdScale->SetResolution(1);
    // this->ThresholdScale->GetEntry()->SetCommandTriggerToAnyChange();

    this->ThresholdScale->SetParent(this->Frame->GetFrame());
    this->ThresholdScale->Create();
    this->ThresholdScale->SetRange(0.0, 300.0);
    this->ThresholdScale->SetMinimumValue(0.0);
    this->ThresholdScale->ClampMinimumValueOn(); 
    this->ThresholdScale->SetMaximumValue(300.0);
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

  }
  this->Script( "pack %s -side top -anchor nw -padx 2 -pady 2", this->ThresholdScale->GetWidgetName());
  // this->Script("grid %s -column 0 -row 0 -sticky nw -padx 2 -pady 2", this->ThresholdScale->GetWidgetName());
}


//----------------------------------------------------------------------------
void vtkTumorGrowthSegmentationStep::ThresholdChangedCallback(
  vtkIdType sel_vol_id, double value)
{
  // Threshold has changed because of user interaction 
  vtkMRMLTumorGrowthNode *mrmlNode = this->GetGUI()->GetNode();
  cout << "vtkTumorGrowthSegmentationStep::SegmentationGlobalPriorChangedCallback ---- Please set correctly" << endl;
}

//----------------------------------------------------------------------------
void vtkTumorGrowthSegmentationStep::TransitionCallback() 
{
   vtkKWWizardWorkflow *wizard_workflow = this->GUI->GetWizardWidget()->GetWizardWorkflow();
   wizard_workflow->AttemptToGoToNextStep();
}


//----------------------------------------------------------------------------
void vtkTumorGrowthSegmentationStep::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
