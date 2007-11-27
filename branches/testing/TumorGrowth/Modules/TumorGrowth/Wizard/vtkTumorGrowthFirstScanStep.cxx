#include "vtkTumorGrowthFirstScanStep.h"
#include "vtkTumorGrowthGUI.h"
#include "vtkKWFrameWithLabel.h"
#include "vtkKWWizardWidget.h"
#include "vtkKWWizardWorkflow.h" 
#include "vtkSlicerNodeSelectorWidget.h"
#include "vtkKWMessageDialog.h"
#include "vtkMRMLTumorGrowthNode.h"

#include "vtkSlicerSliceControllerWidget.h"

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkTumorGrowthFirstScanStep);
vtkCxxRevisionMacro(vtkTumorGrowthFirstScanStep, "$Revision: 1.0 $");

//----------------------------------------------------------------------------
vtkTumorGrowthFirstScanStep::vtkTumorGrowthFirstScanStep()
{
  this->SetName("1/4. Define First Scan");
  this->SetDescription("Select first scan of patient.");
  this->WizardGUICallbackCommand->SetCallback(vtkTumorGrowthFirstScanStep::WizardGUICallback);
}

//----------------------------------------------------------------------------
vtkTumorGrowthFirstScanStep::~vtkTumorGrowthFirstScanStep() { 
}


void vtkTumorGrowthFirstScanStep::UpdateMRML() 
{
  vtkMRMLTumorGrowthNode* node = this->GetGUI()->GetNode();

  if (!node) return;
  if (this->VolumeMenuButton && this->VolumeMenuButton->GetSelected() ) 
  {
    node->SetFirstScanRef(this->VolumeMenuButton->GetSelected()->GetID());
  }
}

void vtkTumorGrowthFirstScanStep::UpdateGUI() {
  vtkMRMLTumorGrowthNode* n = this->GetGUI()->GetNode();
  if (n != NULL &&  this->VolumeMenuButton)
  {
    this->VolumeMenuButton->SetSelected(this->VolumeMenuButton->GetMRMLScene()->GetNodeByID(n->GetFirstScanRef()));
  }
} 

//----------------------------------------------------------------------------
void vtkTumorGrowthFirstScanStep::ShowUserInterface()
{
  this->vtkTumorGrowthSelectScanStep::ShowUserInterface();

  this->Frame->SetLabelText("First Scan");
  this->Script("pack %s -side top -anchor nw -fill x -padx 0 -pady 2", this->Frame->GetWidgetName());

  this->VolumeMenuButton->SetBalloonHelpString("Select first scan of patient.");

  this->Script( "pack %s -side top -anchor nw -padx 2 -pady 2",  this->VolumeMenuButton->GetWidgetName());

  {
    vtkKWWizardWidget *wizard_widget = this->GetGUI()->GetWizardWidget();  
    wizard_widget->BackButtonVisibilityOff();
    wizard_widget->GetCancelButton()->EnabledOff();
  }

}

void vtkTumorGrowthFirstScanStep::WizardGUICallback(vtkObject *caller, unsigned long event, void *clientData, void *callData )
{
    vtkTumorGrowthFirstScanStep *self = reinterpret_cast<vtkTumorGrowthFirstScanStep *>(clientData);
    if( (event == vtkSlicerNodeSelectorWidget::NodeSelectedEvent) && self) {
      self->ProcessGUIEvents(caller, callData);   
    }
}

void vtkTumorGrowthFirstScanStep::ProcessGUIEvents(vtkObject *caller, void *callData) {
    // This just has to be donw if you use the same Callbakc function for severall calls 
    vtkSlicerNodeSelectorWidget *selector = vtkSlicerNodeSelectorWidget::SafeDownCast(caller);

    if (this->VolumeMenuButton && (selector == this->VolumeMenuButton)) 
    { 
      this->TransitionCallback(0);
    }
}


//----------------------------------------------------------------------------
void vtkTumorGrowthFirstScanStep::TransitionCallback(int Flag) 
{
   if (!this->VolumeMenuButton) return;


   vtkKWWizardWidget *wizard_widget = this->GetGUI()->GetWizardWidget();
   vtkKWWizardWorkflow *wizard_workflow = wizard_widget->GetWizardWorkflow();

   if (this->VolumeMenuButton->GetSelected()) { 
     wizard_widget->GetCancelButton()->EnabledOn();
     vtkTumorGrowthGUI *GUI = this->GetGUI();

     vtkSlicerApplicationGUI *AppGUI = GUI->GetApplicationGUI();

     // .. MainSliceGUI0->GetSliceController() =  vtkSlicerSliceControllerWidget
     // .. GetMainSliceGUI0()->GetSliceController()->GetForegroundSelector() = vtkSlicerNodeSelectorWidget
      AppGUI->GetMainSliceGUI0()->GetSliceController()->GetForegroundSelector()->SetSelected(this->VolumeMenuButton->GetSelected());
      AppGUI->GetMainSliceGUI0()->GetSliceController()->GetBackgroundSelector()->SetSelected(this->VolumeMenuButton->GetSelected());
      AppGUI->GetMainSliceGUI1()->GetSliceController()->GetForegroundSelector()->ProcessCommand("None");
      AppGUI->GetMainSliceGUI1()->GetSliceController()->GetBackgroundSelector()->ProcessCommand("None");
      AppGUI->GetMainSliceGUI2()->GetSliceController()->GetForegroundSelector()->ProcessCommand("None");
      AppGUI->GetMainSliceGUI2()->GetSliceController()->GetBackgroundSelector()->ProcessCommand("None");

      // Update entire MRML node bc if it was deleted should be created 
      GUI->UpdateMRML(); 

      // wizard_workflow->AttemptToGoToNextStep();
      // Jumps over two steps ! 

   } else {
     if (Flag) {
       vtkKWMessageDialog::PopupMessage(this->GetGUI()->GetApplication(), this->GetGUI()->GetApplicationGUI()->GetMainSlicerWindow(),"Tumor Growth", "Please define first scan before proceeding", vtkKWMessageDialog::ErrorIcon);
     }
     wizard_widget->GetCancelButton()->EnabledOff();
   }
   cout << "Debugging:vtkTumorGrowthFirstScanStep::TransitionCallback Start" << endl;
   wizard_widget->GetCancelButton()->EnabledOn();
   wizard_workflow->AttemptToGoToNextStep();
   // cout << "vtkTumorGrowthFirstScanStep::TransitionCallback End" << endl;

}

//----------------------------------------------------------------------------
void vtkTumorGrowthFirstScanStep::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
