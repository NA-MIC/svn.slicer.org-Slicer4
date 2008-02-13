#include "vtkTumorGrowthStep.h"
#include "vtkTumorGrowthGUI.h"
#include "vtkMRMLTumorGrowthNode.h"

#include "vtkKWWizardWidget.h"
#include "vtkKWWizardWorkflow.h"
#include "vtkKWFrameWithLabel.h"
#include "vtkCallbackCommand.h"
#include "vtkKWPushButton.h"
#include "vtkSlicerApplication.h"
#include "vtkSlicerVolumesLogic.h" 
#include "vtkSlicerVolumesGUI.h" 
#include "vtkTumorGrowthLogic.h"

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkTumorGrowthStep);
vtkCxxRevisionMacro(vtkTumorGrowthStep, "$Revision: 1.2 $");
vtkCxxSetObjectMacro(vtkTumorGrowthStep,GUI,vtkTumorGrowthGUI);

//----------------------------------------------------------------------------
vtkTumorGrowthStep::vtkTumorGrowthStep()
{
  this->GUI = NULL;
  this->Frame           = NULL;
  this->NextStep = NULL; 
  this->WizardGUICallbackCommand = vtkCallbackCommand::New();
  this->WizardGUICallbackCommand->SetClientData(reinterpret_cast<void *>(this));
  this->GridButton = NULL;
}

//----------------------------------------------------------------------------
vtkTumorGrowthStep::~vtkTumorGrowthStep()
{
  this->SetGUI(NULL);
  if (this->Frame)
  {
    this->Frame->Delete();
    this->Frame = NULL;
  }

  if(this->WizardGUICallbackCommand) 
  {
        this->WizardGUICallbackCommand->Delete();
        this->WizardGUICallbackCommand=NULL;
  }

  if (this->GridButton)
    {
      this->GridButton->Delete();
      this->GridButton = NULL;
    }
}

//----------------------------------------------------------------------------
void vtkTumorGrowthStep::HideUserInterface()
{
  this->Superclass::HideUserInterface();

  if (this->GetGUI())
    {
    this->GetGUI()->GetWizardWidget()->ClearPage();
    }
}

//----------------------------------------------------------------------------
void vtkTumorGrowthStep::Validate()
{
  this->Superclass::Validate();

  vtkKWWizardWorkflow *wizard_workflow = 
    this->GetGUI()->GetWizardWidget()->GetWizardWorkflow();

  wizard_workflow->PushInput(vtkKWWizardStep::GetValidationSucceededInput());
  wizard_workflow->ProcessInputs();
}

//----------------------------------------------------------------------------
int vtkTumorGrowthStep::CanGoToSelf()
{
  return this->Superclass::CanGoToSelf() || 1;
}

void vtkTumorGrowthStep::ShowUserInterface()
{
  this->Superclass::ShowUserInterface();
  
  if (this->NextStep) { this->NextStep->RemoveResults(); }


  vtkKWWizardWidget *wizard_widget = this->GetGUI()->GetWizardWidget();
     wizard_widget->GetCancelButton()->SetEnabled(0);
  vtkKWWidget *parent = wizard_widget->GetClientArea();

  if (!this->Frame)
  {
    this->Frame = vtkKWFrameWithLabel::New();
  }
  if (!this->Frame->IsCreated())
    {
    this->Frame->SetParent(parent);
    this->Frame->Create();
    this->Frame->AllowFrameToCollapseOff();
  }

  wizard_widget->NextButtonVisibilityOff();
  wizard_widget->CancelButtonVisibilityOn();
  wizard_widget->GetCancelButton()->SetText("Next >");
  wizard_widget->GetCancelButton()->SetCommand(this, "TransitionCallback");
  wizard_widget->GetCancelButton()->EnabledOn();

  // Does not work 
  // wizard_widget->GetBackButton()->SetCommand(this, "TransitionToPreviousStep");
  // OK Button only is shown at the end  
}
//----------------------------------------------------------------------------
void vtkTumorGrowthStep::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}


void  vtkTumorGrowthStep::GridCallback() {
  vtkMRMLTumorGrowthNode* Node = this->GetGUI()->GetNode();
  if (!Node) return;
 
  vtkMRMLScalarVolumeNode* currentNode =  vtkMRMLScalarVolumeNode::SafeDownCast(Node->GetScene()->GetNodeByID(Node->GetGrid_Ref()));
  if (currentNode) {
    this->GridRemove();
    this->GridButton->SetReliefToRidge();
  }
  else if (this->GridDefine()) {
    this->GridButton->SetReliefToSunken();
  }
}

void vtkTumorGrowthStep::CreateGridButton() {
  // Grid Button 
  if (!this->GridButton) {
     this->GridButton = vtkKWPushButton::New();
  }

  if (!this->GridButton->IsCreated()) {
    vtkKWWizardWidget *wizard_widget = this->GetGUI()->GetWizardWidget();
    this->GridButton->SetParent(wizard_widget->GetCancelButton()->GetParent());
    this->GridButton->Create();
    this->GridButton->SetWidth(wizard_widget->GetCancelButton()->GetWidth());
    this->GridButton->SetCommand(this, "GridCallback"); 
    this->GridButton->SetText("Grid");
  }
  this->Script("pack %s -side left -anchor nw -expand n -padx 0 -pady 2", this->GridButton->GetWidgetName()); 

  // Button is hold down if Grid already exists 
  vtkMRMLTumorGrowthNode* Node = this->GetGUI()->GetNode();
  if (!Node) return;

  vtkMRMLScalarVolumeNode* currentNode =  vtkMRMLScalarVolumeNode::SafeDownCast(Node->GetScene()->GetNodeByID(Node->GetGrid_Ref()));
  if (currentNode) {
    this->GridButton->SetReliefToSunken(); 
  }
}
  
void vtkTumorGrowthStep::GridRemove() {
  vtkMRMLTumorGrowthNode* Node = this->GetGUI()->GetNode();
  if (Node) {
    vtkMRMLScalarVolumeNode* currentNode =  vtkMRMLScalarVolumeNode::SafeDownCast(Node->GetScene()->GetNodeByID(Node->GetGrid_Ref()));
    if (currentNode) this->GetGUI()->GetMRMLScene()->RemoveNode(currentNode); 
    vtkSlicerApplicationLogic *applicationLogic = this->GetGUI()->GetLogic()->GetApplicationLogic();
    applicationLogic->GetSelectionNode()->SetReferenceActiveLabelVolumeID(NULL);
    applicationLogic->PropagateVolumeSelection();
    Node->SetGrid_Ref(NULL);
  }
}


int vtkTumorGrowthStep::GridDefine() {
  // Initialize
  this->GridRemove();

  vtkMRMLTumorGrowthNode* Node = this->GetGUI()->GetNode();
  if (!Node) return 0 ;

  vtkMRMLScene* mrmlScene       =  Node->GetScene();
  vtkMRMLNode* mrmlScan1Node  =  mrmlScene->GetNodeByID(Node->GetScan1_Ref());
  vtkMRMLVolumeNode* volumeNode =  vtkMRMLVolumeNode::SafeDownCast(mrmlScan1Node);
  if (!volumeNode) return 0;
  
  vtkSlicerApplication    *application   = vtkSlicerApplication::SafeDownCast(this->GetApplication());
  vtkSlicerVolumesGUI     *volumesGUI    = vtkSlicerVolumesGUI::SafeDownCast(application->GetModuleGUIByName("Volumes")); 
  vtkSlicerVolumesLogic   *volumesLogic  = volumesGUI->GetLogic();
  vtkMRMLScalarVolumeNode *GridNode      = volumesLogic->CreateLabelVolume(mrmlScene,volumeNode, "TG_Grid");
  Node->SetGrid_Ref(GridNode->GetID());

  vtkSlicerApplicationLogic *applicationLogic = this->GetGUI()->GetLogic()->GetApplicationLogic();
  applicationLogic->GetSelectionNode()->SetReferenceActiveLabelVolumeID(GridNode->GetID());
  applicationLogic->PropagateVolumeSelection();
  return 1;
}
