#include "vtkProstateNavTargetingStep.h"

#include "vtkProstateNavGUI.h"
#include "vtkProstateNavLogic.h"

#include "vtkKWFrame.h"
#include "vtkKWFrameWithLabel.h"
#include "vtkKWWizardWidget.h"
#include "vtkKWWizardWorkflow.h"

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkProstateNavTargetingStep);
vtkCxxRevisionMacro(vtkProstateNavTargetingStep, "$Revision: 1.1 $");

//----------------------------------------------------------------------------
vtkProstateNavTargetingStep::vtkProstateNavTargetingStep()
{
  this->SetName("4/5. Targeting");
  this->SetDescription("Set target points.");

}

//----------------------------------------------------------------------------
vtkProstateNavTargetingStep::~vtkProstateNavTargetingStep()
{
}

//----------------------------------------------------------------------------
void vtkProstateNavTargetingStep::ShowUserInterface()
{
  this->Superclass::ShowUserInterface();

  vtkKWWizardWidget *wizardWidget = this->GetGUI()->GetWizardWidget();
  wizardWidget->GetCancelButton()->SetEnabled(0);

  vtkKWWidget *parent = wizardWidget->GetClientArea();

  /*
  if (!this->MainFrame)
    {
    this->MainFrame = vtkKWFrame::New();
    this->MainFrame->SetParent(parent);
    this->MainFrame->Create();
    }

  this->Script( "pack %s -side top -anchor nw -expand n -padx 2 -pady 2",
                this->MainFrame->GetWidgetName());
    
  if (!this->TargetListFrame)
    {
    this->TargetListFrame = vtkKWFrame::New();
    this->TargetListFrame->SetParent(this->MainFrame);
    this->TargetListFrame->Create();
    }
  */
}

//----------------------------------------------------------------------------
void vtkProstateNavTargetingStep::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}

//----------------------------------------------------------------------------
void vtkProstateNavTargetingStep::ProcessGUIEvents(vtkObject *caller,
                                          unsigned long event, void *callData)
{
}

