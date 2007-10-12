#include "vtkProstateNavStep.h"
#include "vtkProstateNavGUI.h"

#include "vtkKWWizardWidget.h"
#include "vtkKWWizardWorkflow.h"

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkProstateNavStep);
vtkCxxRevisionMacro(vtkProstateNavStep, "$Revision: 1.2 $");
vtkCxxSetObjectMacro(vtkProstateNavStep,GUI,vtkProstateNavGUI);

//----------------------------------------------------------------------------
vtkProstateNavStep::vtkProstateNavStep()
{
  this->GUI = NULL;
}

//----------------------------------------------------------------------------
vtkProstateNavStep::~vtkProstateNavStep()
{
  this->SetGUI(NULL);
}

//----------------------------------------------------------------------------
void vtkProstateNavStep::HideUserInterface()
{
  this->Superclass::HideUserInterface();

  if (this->GetGUI())
    {
    this->GetGUI()->GetWizardWidget()->ClearPage();
    }
}

//----------------------------------------------------------------------------
void vtkProstateNavStep::Validate()
{
  this->Superclass::Validate();

  vtkKWWizardWorkflow *wizardWorkflow = 
    this->GetGUI()->GetWizardWidget()->GetWizardWorkflow();

  wizardWorkflow->PushInput(vtkKWWizardStep::GetValidationSucceededInput());
  wizardWorkflow->ProcessInputs();
}

//----------------------------------------------------------------------------
int vtkProstateNavStep::CanGoToSelf()
{
  return this->Superclass::CanGoToSelf() || 1;
}

//----------------------------------------------------------------------------
void vtkProstateNavStep::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
