#include "vtkTumorGrowthFirstScanStep.h"
#include "vtkTumorGrowthGUI.h"
#include "vtkKWFrameWithLabel.h"
#include "vtkKWWizardWidget.h"
#include "vtkKWWizardWorkflow.h" 
#include "vtkSlicerNodeSelectorWidget.h"
#include "vtkKWMessageDialog.h"
#include "vtkMRMLTumorGrowthNode.h"
#include "vtkTumorGrowthLogic.h"
#include "vtkSlicerSliceControllerWidget.h"

#include "vtkSlicerVolumesLogic.h"
#include "vtkSlicerVolumesGUI.h"
#include "vtkSlicerApplication.h" 
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
    node->SetScan1_Ref(this->VolumeMenuButton->GetSelected()->GetID());
    // Kilian Set working directory - cut off file name 
    // node->SetWorkgingDir(vtkMRMLVolumeNode::SafeDownCast(this->VolumeMenuButton->GetSelected()->GetID())->GetStorageNode()->GetFileName());
    node->SetWorkingDir("blub");
  }
}

void vtkTumorGrowthFirstScanStep::UpdateGUI() {
  vtkMRMLTumorGrowthNode* n = this->GetGUI()->GetNode();
  if (n != NULL &&  this->VolumeMenuButton)
  {
    this->VolumeMenuButton->SetSelected(this->VolumeMenuButton->GetMRMLScene()->GetNodeByID(n->GetScan1_Ref()));
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

  cout << "DEBUGGING" << endl;
  vtkSlicerApplicationGUI *applicationGUI = this->GetGUI()->GetApplicationGUI();
  if (!applicationGUI) return; 
  char fileName[1024] = "/home/pohl/Slicer/Slicer3-build/blub.mrml";

  std::string fl(fileName);
  applicationGUI->GetMRMLScene()->SetURL(fileName);
  applicationGUI->GetMRMLScene()->Connect();
  this->VolumeMenuButton->SetSelected(applicationGUI->GetMRMLScene()->GetNodeByID("vtkMRMLScalarVolumeNode1")); 
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
      vtkMRMLTumorGrowthNode* node = this->GetGUI()->GetNode();
      if (!node) {
         // Create Node 
         this->GetGUI()->UpdateMRML();
      } else {
         this->UpdateMRML();
      }
      this->TransitionCallback(0);
    }
}


//----------------------------------------------------------------------------
void vtkTumorGrowthFirstScanStep::TransitionCallback(int Flag) 
{
   if (!this->VolumeMenuButton) return;


   vtkKWWizardWidget *wizard_widget = this->GetGUI()->GetWizardWidget();

   if (this->VolumeMenuButton->GetSelected()) { 
     
     wizard_widget->GetCancelButton()->EnabledOn();
     wizard_widget->GetWizardWorkflow()->AttemptToGoToNextStep();
   } else {
     if (Flag) {
       vtkKWMessageDialog::PopupMessage(this->GetGUI()->GetApplication(), this->GetGUI()->GetApplicationGUI()->GetMainSlicerWindow(),"Tumor Growth", "Please define first scan before proceeding", vtkKWMessageDialog::ErrorIcon);
     }
     wizard_widget->GetCancelButton()->EnabledOff();
   }
}

//----------------------------------------------------------------------------
void vtkTumorGrowthFirstScanStep::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
