#include "vtkProstateNavScanControlStep.h"

#include "vtkProstateNavGUI.h"
#include "vtkProstateNavLogic.h"

#include "vtkKWFrame.h"
#include "vtkKWFrameWithLabel.h"
#include "vtkKWWizardWidget.h"
#include "vtkKWWizardWorkflow.h"

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkProstateNavScanControlStep);
vtkCxxRevisionMacro(vtkProstateNavScanControlStep, "$Revision: 1.1 $");

//----------------------------------------------------------------------------
vtkProstateNavScanControlStep::vtkProstateNavScanControlStep()
{
  this->SetName("2/5. Set Scanner Parameters");
  this->SetDescription("Operate the MRI scanner.");

}

//----------------------------------------------------------------------------
vtkProstateNavScanControlStep::~vtkProstateNavScanControlStep()
{
}

//----------------------------------------------------------------------------
void vtkProstateNavScanControlStep::ShowUserInterface()
{
  this->Superclass::ShowUserInterface();

  vtkKWWizardWidget *wizardWidget = this->GetGUI()->GetWizardWidget();
  vtkKWWidget *parent = wizardWidget->GetClientArea();
}

//----------------------------------------------------------------------------
void vtkProstateNavScanControlStep::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}

//----------------------------------------------------------------------------
void vtkProstateNavScanControlStep::ProcessGUIEvents(vtkObject *caller,
                                         unsigned long event, void *callData)
{
}
