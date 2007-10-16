#include "vtkProstateNavCalibrationStep.h"

#include "vtkProstateNavGUI.h"
#include "vtkProstateNavLogic.h"

#include "vtkKWFrame.h"
#include "vtkKWFrameWithLabel.h"
#include "vtkKWWizardWidget.h"
#include "vtkKWWizardWorkflow.h"

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkProstateNavCalibrationStep);
vtkCxxRevisionMacro(vtkProstateNavCalibrationStep, "$Revision: 1.1 $");

//----------------------------------------------------------------------------
vtkProstateNavCalibrationStep::vtkProstateNavCalibrationStep()
{
  this->SetName("3/5. Calibration");
  this->SetDescription("Perform Z-frame calibration.");

}

//----------------------------------------------------------------------------
vtkProstateNavCalibrationStep::~vtkProstateNavCalibrationStep()
{
}

//----------------------------------------------------------------------------
void vtkProstateNavCalibrationStep::ShowUserInterface()
{
  this->Superclass::ShowUserInterface();

  vtkKWWizardWidget *wizardWidget = this->GetGUI()->GetWizardWidget();
  wizardWidget->GetCancelButton()->SetEnabled(0);

  vtkKWWidget *parent = wizardWidget->GetClientArea();
}

//----------------------------------------------------------------------------
void vtkProstateNavCalibrationStep::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
