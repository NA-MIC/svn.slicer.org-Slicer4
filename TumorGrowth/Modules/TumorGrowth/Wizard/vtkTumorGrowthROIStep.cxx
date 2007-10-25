#include "vtkTumorGrowthROIStep.h"

#include "vtkTumorGrowthGUI.h"
#include "vtkMRMLTumorGrowthNode.h"

#include "vtkKWWizardWidget.h"
#include "vtkKWWizardWorkflow.h"
#include "vtkKWFrameWithLabel.h"
#include "vtkKWPushButton.h"
#include "vtkKWLabel.h"
#include "vtkKWMatrixWidgetWithLabel.h"
#include "vtkKWMatrixWidget.h"
#include "vtkSlicerModuleCollapsibleFrame.h"
#include "vtkKWMessageDialog.h"

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkTumorGrowthROIStep);
vtkCxxRevisionMacro(vtkTumorGrowthROIStep, "$Revision: 1.2 $");

//----------------------------------------------------------------------------
vtkTumorGrowthROIStep::vtkTumorGrowthROIStep()
{
  this->SetName("2/4. Define Region of Interest"); 
  this->SetDescription("Define ROI by clicking with <ctrl>-left mouse button around the tumor"); 

  this->FrameButtons    = NULL;
  this->FrameBlank    = NULL;
  this->FrameROI        = NULL;
  this->ButtonsShow     = NULL;
  this->ButtonsReset    = NULL;
  this->ROIMinVector                = NULL;
  this->ROIMaxVector                = NULL;
}

//----------------------------------------------------------------------------
vtkTumorGrowthROIStep::~vtkTumorGrowthROIStep()
{
  if (this->FrameButtons)
  {
    this->FrameButtons->Delete();
    this->FrameButtons = NULL;
  }

  if (this->FrameBlank)
  {
    this->FrameBlank->Delete();
    this->FrameBlank = NULL;
  }

  if (this->FrameROI)
  {
    this->FrameROI->Delete();
    this->FrameROI = NULL;
  }

  if (this->ROIMinVector)
  {
    this->ROIMinVector->Delete();
    this->ROIMinVector = NULL;
  }

  if (this->ROIMaxVector )
  {
    this->ROIMaxVector ->Delete();
    this->ROIMaxVector  = NULL;
  }

}

//----------------------------------------------------------------------------
void vtkTumorGrowthROIStep::ShowUserInterface()
{
  this->vtkTumorGrowthStep::ShowUserInterface();

  // Create the frame
  // Needs to be check bc otherwise with wizrd can be created over again

  this->Frame->SetLabelText("Define ROI");
  this->Script("pack %s -side top -anchor nw -fill x -padx 0 -pady 2", this->Frame->GetWidgetName());

  if (!this->FrameButtons)
    {
    this->FrameButtons = vtkKWFrame::New();
    }
  if (!this->FrameButtons->IsCreated())
    {
      this->FrameButtons->SetParent(this->Frame->GetFrame());
    this->FrameButtons->Create();
    // this->FrameButtons->SetLabelText("");
    // define buttons 
  }
  this->Script("pack %s -side top -anchor nw -fill x -padx 0 -pady 0", this->FrameButtons->GetWidgetName());

  if (!this->FrameBlank)
    {
    this->FrameBlank = vtkKWFrame::New();
    }
  if (!this->FrameBlank->IsCreated())
    {
      this->FrameBlank->SetParent(this->Frame->GetFrame());
    this->FrameBlank->Create();
    // this->FrameButtons->SetLabelText("");
    // define buttons 
  }
  this->Script("pack %s -side top -anchor nw -fill x -padx 0 -pady 4", this->FrameBlank->GetWidgetName());

  if (!this->FrameROI)
    {
    this->FrameROI = vtkSlicerModuleCollapsibleFrame::New();
    }
  if (!this->FrameROI->IsCreated())
    {
      this->FrameROI->SetParent(this->Frame->GetFrame());
    this->FrameROI->Create();
    this->FrameROI->SetLabelText("Manual");
    this->FrameROI->CollapseFrame();
  }

  this->Script("pack %s -side top -anchor nw -fill x -padx 0 -pady 0", this->FrameROI->GetWidgetName());

  if (!this->ButtonsShow) {
    this->ButtonsShow = vtkKWPushButton::New();
  }

  if (!this->ButtonsShow->IsCreated()) {
    this->ButtonsShow->SetParent(this->FrameButtons);
    this->ButtonsShow->Create();
    this->ButtonsShow->SetWidth(TUMORGROWTH_MENU_BUTTON_WIDTH);
    this->ButtonsShow->SetText("Show ROI");
  }

  if (!this->ButtonsReset) {
    this->ButtonsReset = vtkKWPushButton::New();
  }
  if (!this->ButtonsReset->IsCreated()) {
    this->ButtonsReset->SetParent(this->FrameButtons);
    this->ButtonsReset->Create();
    this->ButtonsReset->SetWidth(TUMORGROWTH_MENU_BUTTON_WIDTH);
    this->ButtonsReset->SetText("Reset");
  }

  this->Script("pack %s %s -side left -anchor nw -expand n -padx 2 -pady 2", 
                this->ButtonsShow->GetWidgetName(),this->ButtonsReset->GetWidgetName());

  if (!this->ROIMaxVector)
    {
    this->ROIMaxVector = vtkKWMatrixWidgetWithLabel::New();
    }
  if (!this->ROIMaxVector->IsCreated())
    {
      this->ROIMaxVector->SetParent(this->FrameROI->GetFrame());
    this->ROIMaxVector->Create();
    this->ROIMaxVector->SetLabelText("Max :");
    this->ROIMaxVector->SetLabelPositionToLeft();
    this->ROIMaxVector->ExpandWidgetOff();
    this->ROIMaxVector->GetLabel()->SetWidth(TUMORGROWTH_WIDGETS_LABEL_WIDTH - 16);
    this->ROIMaxVector->SetBalloonHelpString("Set the upper right hand corner of region of interest.");
    
    vtkKWMatrixWidget *matrix = this->ROIMaxVector->GetWidget();
      matrix->SetNumberOfColumns(3);
      matrix->SetNumberOfRows(1);
      matrix->SetElementWidth(4);
      matrix->SetRestrictElementValueToInteger();
      matrix->SetElementChangedCommand(this, "ROIMaxChangedCallback");
      matrix->SetElementChangedCommandTriggerToAnyChange();
    }
    // Set it up so it has default value from MRML file 

  if (!this->ROIMinVector)
    {
    this->ROIMinVector = vtkKWMatrixWidgetWithLabel::New();
    }
  if (!this->ROIMinVector->IsCreated())
    {
    this->ROIMinVector->SetParent(this->FrameROI->GetFrame());
    this->ROIMinVector->Create();
    this->ROIMinVector->SetLabelText("Min :");
    this->ROIMinVector->SetLabelPositionToLeft();
    this->ROIMinVector->ExpandWidgetOff();
    this->ROIMinVector->GetLabel()->SetWidth(TUMORGROWTH_WIDGETS_LABEL_WIDTH - 16);
    this->ROIMinVector->SetBalloonHelpString("Set the upper right hand corner of region of interest.");
    
    vtkKWMatrixWidget *matrix = this->ROIMinVector->GetWidget();
      matrix->SetNumberOfColumns(3);
      matrix->SetNumberOfRows(1);
      matrix->SetElementWidth(4);
      matrix->SetRestrictElementValueToInteger();
      matrix->SetElementChangedCommand(this, "ROIMinChangedCallback");
      matrix->SetElementChangedCommandTriggerToAnyChange();
    }
    // Set it up so it has default value from MRML file 
  this->Script("pack %s %s -side top -anchor nw -padx 2 -pady 2",this->ROIMinVector->GetWidgetName(),this->ROIMaxVector->GetWidgetName());
  {
   vtkKWWizardWidget *wizard_widget = this->GetGUI()->GetWizardWidget(); 
   wizard_widget->BackButtonVisibilityOn();
   wizard_widget->GetCancelButton()->EnabledOn();
  }
}


//----------------------------------------------------------------------------
void vtkTumorGrowthROIStep::ROIMaxChangedCallback(
  vtkIdType sel_vol_id, double value)
{
  // Threshold has changed because of user interaction 
  vtkMRMLTumorGrowthNode *mrmlNode = this->GetGUI()->GetNode();
  cout << "vtkTumorGrowthROIStep::ROIMaxChangedCallback ---- Please set correctly" << endl;
}

//----------------------------------------------------------------------------
void vtkTumorGrowthROIStep::ROIMinChangedCallback(
  vtkIdType sel_vol_id, double value)
{
  // Threshold has changed because of user interaction 
  vtkMRMLTumorGrowthNode *mrmlNode = this->GetGUI()->GetNode();
  cout << "vtkTumorGrowthROIStep::ROIMinChangedCallback ---- Please set correctly" << endl;
}



//----------------------------------------------------------------------------
void vtkTumorGrowthROIStep::TransitionCallback() 
{
   vtkKWWizardWorkflow *wizard_workflow = this->GUI->GetWizardWidget()->GetWizardWorkflow();
 
   // Put in check 
   if (1) { 
     cout << "----Debugging:vtkTumorGrowthROIStep::TransitionCallback " << endl;
     wizard_workflow->AttemptToGoToNextStep();
   } else {
     vtkKWMessageDialog::PopupMessage(this->GUI->GetApplication(), this->GUI->GetApplicationGUI()->GetMainSlicerWindow(),"Tumor Growth",
           "Please define first scan before proceeding", vtkKWMessageDialog::ErrorIcon);
   }
}

//----------------------------------------------------------------------------
void vtkTumorGrowthROIStep::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
