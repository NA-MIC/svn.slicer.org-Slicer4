#include "vtkTumorGrowthSecondScanStep.h"
#include "vtkTumorGrowthGUI.h"
#include "vtkKWFrameWithLabel.h"
#include "vtkKWWizardWidget.h"
#include "vtkKWWizardWorkflow.h"
#include "vtkSlicerNodeSelectorWidget.h"
#include "vtkKWMessageDialog.h"
#include "vtkMRMLTumorGrowthNode.h"

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkTumorGrowthSecondScanStep);
vtkCxxRevisionMacro(vtkTumorGrowthSecondScanStep, "$Revision: 1.0 $");

//----------------------------------------------------------------------------
vtkTumorGrowthSecondScanStep::vtkTumorGrowthSecondScanStep()
{
  this->SetName("4/4. Define Second Scan");
  this->SetDescription("Select second scan of patient.");
  this->WizardGUICallbackCommand->SetCallback(vtkTumorGrowthSecondScanStep::WizardGUICallback);
}

//----------------------------------------------------------------------------
vtkTumorGrowthSecondScanStep::~vtkTumorGrowthSecondScanStep() { }

//----------------------------------------------------------------------------
void vtkTumorGrowthSecondScanStep::ShowUserInterface()
{
  this->vtkTumorGrowthSelectScanStep::ShowUserInterface();

  this->Frame->SetLabelText("Second Scan");
  this->VolumeMenuButton->SetBalloonHelpString("Select first scan of patient.");

  this->Script("pack %s -side top -anchor nw -fill x -padx 0 -pady 2", this->Frame->GetWidgetName());
  this->Script( "pack %s -side top -anchor nw -padx 2 -pady 2",  this->VolumeMenuButton->GetWidgetName());

  {
    vtkKWWizardWidget *wizard_widget = this->GetGUI()->GetWizardWidget();
    wizard_widget->GetCancelButton()->SetText("Analyze");
    wizard_widget->GetCancelButton()->EnabledOff();
  }

} 

void vtkTumorGrowthSecondScanStep::WizardGUICallback(vtkObject *caller, unsigned long event, void *clientData, void *callData )
{
    vtkTumorGrowthSecondScanStep *self = reinterpret_cast<vtkTumorGrowthSecondScanStep *>(clientData);
    if( (event == vtkSlicerNodeSelectorWidget::NodeSelectedEvent) && self) {
      self->ProcessGUIEvents(caller, callData);   
    }
}

void vtkTumorGrowthSecondScanStep::ProcessGUIEvents(vtkObject *caller, void *callData) {
    // This just has to be donw if you use the same Callbakc function for severall calls 
    vtkSlicerNodeSelectorWidget *selector = vtkSlicerNodeSelectorWidget::SafeDownCast(caller);

    if (this->VolumeMenuButton && (selector == this->VolumeMenuButton) && (this->VolumeMenuButton->GetSelected() != NULL) )
    { 
      this->TransitionCallback(0);
    }
    cout << " >>>>>>>>>>>>>>> Debugging <<<<<<<<" << endl; 
    this->TransitionCallback(0);
}

void vtkTumorGrowthSecondScanStep::UpdateGUI() {
  vtkMRMLTumorGrowthNode* n = this->GetGUI()->GetNode();
  if (n != NULL && this->VolumeMenuButton)
  {
    this->VolumeMenuButton->SetSelected(this->VolumeMenuButton->GetMRMLScene()->GetNodeByID(n->GetSecondScanRef()));
  }
} 

// Call from Cancel button (which is now analyze) or by selecting volume 
void vtkTumorGrowthSecondScanStep::TransitionCallback(int Flag) 
{
  if (!this->VolumeMenuButton) return;

  vtkKWWizardWidget *wizard_widget = this->GetGUI()->GetWizardWidget();
  vtkKWWizardWorkflow *wizard_workflow = wizard_widget->GetWizardWorkflow();

  if (this->VolumeMenuButton->GetSelected()) { 
     wizard_widget->GetCancelButton()->EnabledOn();
     // put in proceccing of images here 
     wizard_workflow->AttemptToGoToNextStep();
  } else {
     if (Flag) {
       vtkKWMessageDialog::PopupMessage(this->GetGUI()->GetApplication(), this->GetGUI()->GetApplicationGUI()->GetMainSlicerWindow(),"Tumor Growth", "Please define scan before proceeding", vtkKWMessageDialog::ErrorIcon);
     }
     wizard_widget->GetCancelButton()->EnabledOff();
  }
  cout << "Debugging:vtkTumorGrowthSelectScanStep::TransitionCallback " << endl;
  wizard_workflow->AttemptToGoToNextStep();
}

//----------------------------------------------------------------------------
void vtkTumorGrowthSecondScanStep::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
