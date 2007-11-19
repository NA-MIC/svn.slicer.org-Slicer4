#include "vtkProstateNavManualControlStep.h"

#include "vtkProstateNavGUI.h"
#include "vtkProstateNavLogic.h"

#include "vtkKWFrame.h"
#include "vtkKWFrameWithLabel.h"
#include "vtkKWWizardWidget.h"
#include "vtkKWWizardWorkflow.h"

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkProstateNavManualControlStep);
vtkCxxRevisionMacro(vtkProstateNavManualControlStep, "$Revision: 1.1 $");

//----------------------------------------------------------------------------
vtkProstateNavManualControlStep::vtkProstateNavManualControlStep()
{
  this->SetName("5/5. Manual Control");
  this->SetDescription("Insert the needle.");

}

//----------------------------------------------------------------------------
vtkProstateNavManualControlStep::~vtkProstateNavManualControlStep()
{
}

//----------------------------------------------------------------------------
void vtkProstateNavManualControlStep::ShowUserInterface()
{
  this->Superclass::ShowUserInterface();

  vtkKWWizardWidget *wizardWidget = this->GetGUI()->GetWizardWidget();
  wizardWidget->GetCancelButton()->SetEnabled(0);

  vtkKWWidget *parent = wizardWidget->GetClientArea();
}

//----------------------------------------------------------------------------
void vtkProstateNavManualControlStep::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}

//----------------------------------------------------------------------------
void vtkProstateNavManualControlStep::ProcessGUIEvents(vtkObject *caller,
                                          unsigned long event, void *callData)
{
}
