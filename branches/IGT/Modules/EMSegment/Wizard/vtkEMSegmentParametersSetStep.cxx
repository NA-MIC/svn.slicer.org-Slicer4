#include "vtkEMSegmentParametersSetStep.h"

#include "vtkEMSegmentGUI.h"
#include "vtkEMSegmentLogic.h"

#include "vtkKWWizardWidget.h"
#include "vtkKWWizardWorkflow.h"
#include "vtkKWFrame.h"
#include "vtkKWFrameWithLabel.h"
#include "vtkKWLabel.h"
#include "vtkKWMenu.h"
#include "vtkKWMenuButton.h"
#include "vtkKWMenuButtonWithLabel.h"
#include "vtkKWTreeWithScrollbars.h"
#include "vtkKWTree.h"

#include "vtkEMSegmentAnatomicalStructureStep.h"

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkEMSegmentParametersSetStep);
vtkCxxRevisionMacro(vtkEMSegmentParametersSetStep, "$Revision: 1.2 $");

//----------------------------------------------------------------------------
vtkEMSegmentParametersSetStep::vtkEMSegmentParametersSetStep()
{
  this->SetName("1/8. Define Parameters Set");
  this->SetDescription("Select Parameters Set or create new parameters.");

  this->ParameterSetFrame      = NULL;
  this->ParameterSetMenuButton = NULL;
}

//----------------------------------------------------------------------------
vtkEMSegmentParametersSetStep::~vtkEMSegmentParametersSetStep()
{
  if (this->ParameterSetMenuButton)
    {
    this->ParameterSetMenuButton->Delete();
    this->ParameterSetMenuButton = NULL;
    }

  if (this->ParameterSetFrame)
    {
    this->ParameterSetFrame->Delete();
    this->ParameterSetFrame = NULL;
    }
}

//----------------------------------------------------------------------------
void vtkEMSegmentParametersSetStep::ShowUserInterface()
{
  this->Superclass::ShowUserInterface();

  vtkKWWizardWidget *wizard_widget = this->GetGUI()->GetWizardWidget();

  wizard_widget->GetCancelButton()->SetEnabled(0);

  // Create the Parameters set frame

  if (!this->ParameterSetFrame)
    {
    this->ParameterSetFrame = vtkKWFrameWithLabel::New();
    }
  if (!this->ParameterSetFrame->IsCreated())
    {
    this->ParameterSetFrame->SetParent(wizard_widget->GetClientArea());
    this->ParameterSetFrame->Create();
    this->ParameterSetFrame->SetLabelText("Select Parameters Set");
    }

  this->Script("pack %s -side top -expand n -fill both -padx 0 -pady 2", 
               this->ParameterSetFrame->GetWidgetName());

  // Create the Parameters Set Menu button

  if (!this->ParameterSetMenuButton)
    {
    this->ParameterSetMenuButton = vtkKWMenuButtonWithLabel::New();
    }
  if (!this->ParameterSetMenuButton->IsCreated())
    {
    this->ParameterSetMenuButton->SetParent(
      this->ParameterSetFrame->GetFrame());
    this->ParameterSetMenuButton->Create();
    this->ParameterSetMenuButton->GetLabel()->SetWidth(
      EMSEG_WIDGETS_LABEL_WIDTH - 10);
    this->ParameterSetMenuButton->SetLabelText("Parameters Set:");
    this->ParameterSetMenuButton->GetWidget()->SetWidth(
      EMSEG_MENU_BUTTON_WIDTH + 10);
    this->ParameterSetMenuButton->SetBalloonHelpString(
      "Select Parameters Set.");
    }

  this->Script("pack %s -side top -anchor nw -padx 2 -pady 2", 
               this->ParameterSetMenuButton->GetWidgetName());
  
  this->UpdateLoadedParameterSets();
}

//----------------------------------------------------------------------------
void vtkEMSegmentParametersSetStep::PopulateLoadedParameterSets(
  vtkObject *obj, const char *method)
{
  if(!this->ParameterSetMenuButton ||
     !this->ParameterSetMenuButton->IsCreated())
    {
    return;
    }

  vtkEMSegmentLogic *logic = this->GetGUI()->GetLogic();

  vtkKWMenu *menu = 
    this->ParameterSetMenuButton->GetWidget()->GetMenu();
  menu->DeleteAllItems();
  char buffer[256];
  
  sprintf(buffer, "%s %d", method, -1);
  menu->AddRadioButton("Create New Parameters", obj, buffer);
  
  int nb_of_sets = logic->GetNumberOfParameterSets();
  for(int index = 0; index < nb_of_sets; index++)
    {
    const char *name = logic->GetNthParameterSetName(index);
    if (name)
      {
      sprintf(buffer, "%s %d", method, index);
      menu->AddRadioButton(name, this, buffer);
      }
    }
}

//----------------------------------------------------------------------------
void vtkEMSegmentParametersSetStep::UpdateLoadedParameterSets()
{
  if(!this->ParameterSetMenuButton ||
     !this->ParameterSetMenuButton->IsCreated())
    {
    return;
    }

  vtkEMSegmentLogic *logic = this->GetGUI()->GetLogic();

  vtkKWMenuButton *menuButton = this->ParameterSetMenuButton->GetWidget();
  vtksys_stl::string sel_value = "";
  if(menuButton->GetValue())
    {
    sel_value = menuButton->GetValue();
    }

  this->PopulateLoadedParameterSets(
    this, "SelectedParameterSetChangedCallback");

  if (strcmp(sel_value.c_str(), "") != 0)
    {
    // Select the original
    int nb_of_sets = menuButton->GetMenu()->GetNumberOfItems();
    for (int index = 0; index < nb_of_sets; index++)
      {
      const char *name = menuButton->GetMenu()->GetItemLabel(index);
      if (name && strcmp(sel_value.c_str(), name) == 0)
        {
        menuButton->GetMenu()->SelectItem(index);
        return;
        }
      }
    }

  // if there is no previous selection, select the first loaded set,
  // or if there is no loaded set, leave it blank

  int nb_of_sets = logic->GetNumberOfParameterSets();
  if(nb_of_sets > 0 &&
     menuButton->GetMenu()->GetNumberOfItems() > 1)
    {
    this->ParameterSetMenuButton->GetWidget()->GetMenu()->SelectItem(1);
    this->SelectedParameterSetChangedCallback(0);
    }
}

//----------------------------------------------------------------------------
void vtkEMSegmentParametersSetStep::SelectedParameterSetChangedCallback(
  int index)
{
  vtkEMSegmentLogic *logic = this->GetGUI()->GetLogic();

  // New Parameters

  if (index < 0)
    {
    logic->CreateAndObserveNewParameterSet();
    //Assuming the logic adds the node to the end.
    int nb_of_sets = logic->GetNumberOfParameterSets();
    if (nb_of_sets > 0)
      {
      this->UpdateLoadedParameterSets();
      const char *name = logic->GetNthParameterSetName(nb_of_sets-1);
      if (name)
        {
        // Select the newly created parameter set
        vtkKWMenuButton *menuButton = 
          this->ParameterSetMenuButton->GetWidget();
        if (menuButton->GetMenu()->GetNumberOfItems() == nb_of_sets + 1)
          {
          menuButton->GetMenu()->SelectItem(nb_of_sets);
          }
        }
      }
    }
  else
    {
    logic->SetLoadedParameterSetIndex(index);
    }
  
  vtkEMSegmentAnatomicalStructureStep *anat_step = 
    this->GetGUI()->GetAnatomicalStructureStep();
  if (anat_step && 
      anat_step->GetAnatomicalStructureTree() && 
      anat_step->GetAnatomicalStructureTree()->IsCreated())
    {
    anat_step->GetAnatomicalStructureTree()->GetWidget()->DeleteAllNodes();
    }
}

//----------------------------------------------------------------------------
void vtkEMSegmentParametersSetStep::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
