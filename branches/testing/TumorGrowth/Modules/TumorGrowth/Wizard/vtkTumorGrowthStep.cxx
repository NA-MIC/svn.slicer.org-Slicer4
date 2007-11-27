#include "vtkTumorGrowthStep.h"
#include "vtkTumorGrowthGUI.h"
#include "vtkMRMLTumorGrowthNode.h"

#include "vtkKWWizardWidget.h"
#include "vtkKWWizardWorkflow.h"
#include "vtkKWFrameWithLabel.h"
#include "vtkCallbackCommand.h"

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkTumorGrowthStep);
vtkCxxRevisionMacro(vtkTumorGrowthStep, "$Revision: 1.2 $");
vtkCxxSetObjectMacro(vtkTumorGrowthStep,GUI,vtkTumorGrowthGUI);

//----------------------------------------------------------------------------
vtkTumorGrowthStep::vtkTumorGrowthStep()
{
  this->GUI = NULL;
  this->Frame           = NULL;
  this->WizardGUICallbackCommand = vtkCallbackCommand::New();
  this->WizardGUICallbackCommand->SetClientData(reinterpret_cast<void *>(this));
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
 
}
//----------------------------------------------------------------------------
void vtkTumorGrowthStep::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
