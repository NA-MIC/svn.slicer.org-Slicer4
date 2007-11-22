#include "vtkProstateNavStep.h"
#include "vtkProstateNavGUI.h"
#include "vtkProstateNavLogic.h"

#include "vtkKWWizardWidget.h"
#include "vtkKWWizardWorkflow.h"

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkProstateNavStep);
vtkCxxRevisionMacro(vtkProstateNavStep, "$Revision: 1.2 $");
vtkCxxSetObjectMacro(vtkProstateNavStep,GUI,vtkProstateNavGUI);
vtkCxxSetObjectMacro(vtkProstateNavStep,Logic,vtkProstateNavLogic);

//----------------------------------------------------------------------------
vtkProstateNavStep::vtkProstateNavStep()
{
  this->GUI = NULL;
  this->Logic = NULL;

  GUICallbackCommand = vtkCallbackCommand::New();
  GUICallbackCommand->SetClientData(this);
  GUICallbackCommand->SetCallback(&vtkProstateNavStep::GUICallback);

}

//----------------------------------------------------------------------------
vtkProstateNavStep::~vtkProstateNavStep()
{
  this->SetGUI(NULL);
  this->SetLogic(NULL);
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

//----------------------------------------------------------------------------
void vtkProstateNavStep::ShowUserInterface()
{
  this->Superclass::ShowUserInterface();

  vtkKWWizardWidget *wizardWidget = this->GetGUI()->GetWizardWidget();
  wizardWidget->GetCancelButton()->SetEnabled(0);
  wizardWidget->SetTitleAreaBackgroundColor(this->TitleBackgroundColor[0],
                                            this->TitleBackgroundColor[1],
                                            this->TitleBackgroundColor[2]);

}

//----------------------------------------------------------------------------
void vtkProstateNavStep::GUICallback( vtkObject *caller,
                           unsigned long eid, void *clientData, void *callData )
{

  vtkProstateNavStep *self = reinterpret_cast<vtkProstateNavStep *>(clientData);
  
  if (self->GetInGUICallbackFlag())
    {
#ifdef _DEBUG
    vtkDebugWithObjectMacro(self, "vtkProstateNavStep *********GUICallback called recursively?");
#endif
    }

  vtkDebugWithObjectMacro(self, "In vtkProstateNavStep GUICallback");
  
  self->SetInGUICallbackFlag(1);
  self->ProcessGUIEvents(caller, eid, callData);
  self->SetInGUICallbackFlag(0);
  
}
