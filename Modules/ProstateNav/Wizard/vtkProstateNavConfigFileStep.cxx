#include "vtkProstateNavConfigFileStep.h"

#include "vtkProstateNavGUI.h"
#include "vtkProstateNavLogic.h"

#include "vtkKWFrame.h"
#include "vtkKWFrameWithLabel.h"
#include "vtkKWEntry.h"
#include "vtkKWCheckButton.h"
#include "vtkKWWizardWidget.h"
#include "vtkKWWizardWorkflow.h"
#include "vtkKWLoadSaveButton.h"
#include "vtkKWLoadSaveButtonWithLabel.h"

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkProstateNavConfigFileStep);
vtkCxxRevisionMacro(vtkProstateNavConfigFileStep, "$Revision: 1.1 $");

//----------------------------------------------------------------------------
vtkProstateNavConfigFileStep::vtkProstateNavConfigFileStep()
{
  this->SetName("1/2. Load Configuration File (Not Functional Yet)");
  this->SetDescription("Load the configuration file used by NaviTrack.");

  this->ConfigNTFrame = NULL;
  this->ConnectNTFrame = NULL;
  this->LoadConfigButtonNT = NULL;
  this->ConfigFileEntryNT = NULL;
  this->ConnectCheckButtonNT = NULL;
}

//----------------------------------------------------------------------------
vtkProstateNavConfigFileStep::~vtkProstateNavConfigFileStep()
{
  if (this->LoadConfigButtonNT)
    {
    this->LoadConfigButtonNT->SetParent(NULL );
    this->LoadConfigButtonNT->Delete();
    }
  if (this->ConfigFileEntryNT)
    {
    this->ConfigFileEntryNT->SetParent(NULL );
    this->ConfigFileEntryNT->Delete();
    }
  if (this->ConnectNTFrame)
    {
    this->ConnectNTFrame->SetParent(NULL);
    this->ConnectNTFrame->Delete();
    }
  if (this->ConfigNTFrame)
    {
    this->ConfigNTFrame->SetParent(NULL);
    this->ConfigNTFrame->Delete();
    }
}

//----------------------------------------------------------------------------
void vtkProstateNavConfigFileStep::ShowUserInterface()
{
  this->Superclass::ShowUserInterface();

  vtkKWWizardWidget *wizardWidget = this->GetGUI()->GetWizardWidget();
  wizardWidget->GetCancelButton()->SetEnabled(0);

  vtkKWWidget *parent = wizardWidget->GetClientArea();

  // Create the frame

  if (!this->ConfigNTFrame)
    {
    this->ConfigNTFrame = vtkKWFrame::New();
    this->ConfigNTFrame->SetParent ( parent );
    this->ConfigNTFrame->Create ( );
    }

  this->Script( "pack %s -side top -anchor nw -expand n -padx 2 -pady 2",
                this->ConfigNTFrame->GetWidgetName());
    
  if (!this->ConnectNTFrame)
    {
    this->ConnectNTFrame = vtkKWFrame::New();
    this->ConnectNTFrame->SetParent ( parent );
    this->ConnectNTFrame->Create ( );
    }

  this->Script( "pack %s -side top -anchor nw -expand n -padx 2 -pady 2",
                this->ConnectNTFrame->GetWidgetName());

  // Create the file entry and load button

  if (!this->LoadConfigButtonNT)
    {
    this->LoadConfigButtonNT = vtkKWLoadSaveButtonWithLabel::New();
    this->LoadConfigButtonNT->SetParent(this->ConfigNTFrame);
    this->LoadConfigButtonNT->Create();
    this->LoadConfigButtonNT->SetWidth(15);
    this->LoadConfigButtonNT->GetWidget()->SetText ("Browse Config File");
    this->LoadConfigButtonNT->GetWidget()->GetLoadSaveDialog()->SetFileTypes(
      "{ {ProstateNav} {*.xml} }");
    this->LoadConfigButtonNT->GetWidget()->GetLoadSaveDialog()
      ->RetrieveLastPathFromRegistry("OpenPath");

    this->Script("pack %s -side left -anchor w -fill x -padx 2 -pady 2", 
                 this->LoadConfigButtonNT->GetWidgetName());
    }

  if (!this->ConfigFileEntryNT)
    {
    this->ConfigFileEntryNT = vtkKWEntry::New();
    this->ConfigFileEntryNT->SetParent(this->ConfigNTFrame);
    this->ConfigFileEntryNT->Create();
    this->ConfigFileEntryNT->SetWidth(50);
    this->ConfigFileEntryNT->SetValue ("");
    
    this->Script("pack %s -side left -anchor w -fill x -padx 2 -pady 2", 
                 this->ConfigFileEntryNT->GetWidgetName());
    }

  // The connnect button 
  
  if (!this->ConnectCheckButtonNT)
    {
    this->ConnectCheckButtonNT = vtkKWCheckButton::New();
    this->ConnectCheckButtonNT->SetParent(this->ConnectNTFrame);
    this->ConnectCheckButtonNT->Create();
    this->ConnectCheckButtonNT->SelectedStateOff();
    this->ConnectCheckButtonNT->SetText("Connect");
    this->Script("pack %s -side top -anchor w -padx 2 -pady 2", 
                 this->ConnectCheckButtonNT->GetWidgetName());
    }
}

//----------------------------------------------------------------------------
void vtkProstateNavConfigFileStep::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
