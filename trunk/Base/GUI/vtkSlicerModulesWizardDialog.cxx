#include "vtkSlicerModulesWizardDialog.h"

#include "vtkObjectFactory.h"

#include "vtkSlicerModulesConfigurationStep.h"
#include "vtkSlicerModulesStep.h"
#include "vtkSlicerModulesResultStep.h"

#include "vtkKWApplication.h"
#include "vtkKWWizardStep.h"
#include "vtkKWWizardWidget.h"
#include "vtkKWWizardWorkflow.h"
#include "vtkKWLabel.h"
#include "vtkKWIcon.h"

#include <vtksys/ios/sstream>

//----------------------------------------------------------------------------
vtkStandardNewMacro( vtkSlicerModulesWizardDialog );
vtkCxxRevisionMacro(vtkSlicerModulesWizardDialog, "$Revision: 1.6 $");

//----------------------------------------------------------------------------
vtkSlicerModulesWizardDialog::vtkSlicerModulesWizardDialog()
{
  this->ModulesConfigurationStep    = NULL;
  this->ModulesStep    = NULL;
  this->ModulesResultStep    = NULL;
}

//----------------------------------------------------------------------------
void vtkSlicerModulesWizardDialog::OK()
{
  this->Superclass::OK();

  this->Script("exec $::env(Slicer3_HOME)/Slicer3 &; $::slicer3::Application SetPromptBeforeExit 0; exit");
}

//----------------------------------------------------------------------------
void vtkSlicerModulesWizardDialog::CreateWidget()
{
  // Check if already created

  if (this->IsCreated())
    {
    vtkErrorMacro("class already created");
    return;
    }

  // Call the superclass to create the whole widget

  this->Superclass::CreateWidget();

  vtkKWWizardWorkflow *wizard_workflow = this->GetWizardWorkflow();
  vtkKWWizardWidget *wizard_widget = this->GetWizardWidget();

  wizard_widget->GetTitleIconLabel()->SetImageToPredefinedIcon(
    vtkKWIcon::IconNetDrive);

  // Add Configuration step

  this->ModulesConfigurationStep = vtkSlicerModulesConfigurationStep::New();
  this->ModulesConfigurationStep->SetWizardDialog(this);
  wizard_workflow->AddStep(this->ModulesConfigurationStep);
  this->ModulesConfigurationStep->Delete();

  // Add Modules step

  this->ModulesStep = vtkSlicerModulesStep::New();
  this->ModulesStep->SetWizardDialog(this);
  wizard_workflow->AddNextStep(this->ModulesStep);
  this->ModulesStep->Delete();

  // Add Result step

  this->ModulesResultStep = vtkSlicerModulesResultStep::New();
  this->ModulesResultStep->SetWizardDialog(this);
  wizard_workflow->AddNextStep(this->ModulesResultStep);
  this->ModulesResultStep->Delete();

  // -----------------------------------------------------------------
  // Initial and finish step

  wizard_workflow->SetFinishStep(this->ModulesResultStep);
  wizard_workflow->CreateGoToTransitionsToFinishStep();
  wizard_workflow->SetInitialStep(this->ModulesConfigurationStep);
}
