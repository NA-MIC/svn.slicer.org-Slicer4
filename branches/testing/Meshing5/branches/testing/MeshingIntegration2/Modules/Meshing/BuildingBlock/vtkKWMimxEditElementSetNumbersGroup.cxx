/*=========================================================================

Program:   MIMX Meshing Toolkit
Module:    $RCSfile: vtkKWMimxEditElementSetNumbersGroup.cxx,v $
Language:  C++
Date:      $Date: 2008/04/25 21:31:09 $
Version:   $Revision: 1.1 $

 Musculoskeletal Imaging, Modelling and Experimentation (MIMX)
 Center for Computer Aided Design
 The University of Iowa
 Iowa City, IA 52242
 http://www.ccad.uiowa.edu/mimx/
 
Copyright (c) The University of Iowa. All rights reserved.
See MIMXCopyright.txt or http://www.ccad.uiowa.edu/mimx/Copyright.htm for details.

This software is distributed WITHOUT ANY WARRANTY; without even 
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
PURPOSE.  See the above copyright notices for more information.

=========================================================================*/

#include "vtkKWMimxEditElementSetNumbersGroup.h"
#include "vtkKWMimxMainWindow.h"
#include "vtkKWMimxMainMenuGroup.h"
#include "vtkMimxErrorCallback.h"
#include "vtkKWMimxMainNotebook.h"

#include "vtkLinkedListWrapper.h"

#include "vtkActor.h"
#include "vtkMimxUnstructuredGridActor.h"
#include "vtkMimxMeshActor.h"
#include "vtkUnstructuredGrid.h"

#include "vtkKWApplication.h"
#include "vtkKWEntryWithLabel.h"
#include "vtkKWEntry.h"
#include "vtkKWEvent.h"
#include "vtkKWFrame.h"
#include "vtkKWFrameWithLabel.h"
#include "vtkKWLabel.h"
#include "vtkKWMenu.h"
#include "vtkKWMenuButton.h"
#include "vtkKWMenuButtonWithLabel.h"
#include "vtkKWOptions.h"
#include "vtkKWRenderWidget.h"
#include "vtkKWTkUtilities.h"
#include "vtkObjectFactory.h"
#include "vtkKWComboBoxWithLabel.h"
#include "vtkKWComboBox.h"
#include "vtkKWPushButton.h"
#include "vtkKWMimxMainUserInterfacePanel.h"
#include "vtkFieldData.h"
#include "vtkFloatArray.h"
#include "vtkCellData.h"
#include "vtkIntArray.h"
#include "vtkStringArray.h"
#include "vtkDoubleArray.h"
#include "vtkKWMessageDialog.h"

#include "vtkRenderer.h"

#include <vtksys/stl/list>
#include <vtksys/stl/algorithm>

// define the option types
#define VTK_KW_OPTION_NONE         0
#define VTK_KW_OPTION_LOAD                 1

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkKWMimxEditElementSetNumbersGroup);
vtkCxxRevisionMacro(vtkKWMimxEditElementSetNumbersGroup, "$Revision: 1.1 $");

//----------------------------------------------------------------------------
vtkKWMimxEditElementSetNumbersGroup::vtkKWMimxEditElementSetNumbersGroup()
{
  this->ObjectListComboBox = NULL;
  this->ElementSetComboBox = NULL;
  this->StartingElementNumberEntry = NULL;
}

//----------------------------------------------------------------------------
vtkKWMimxEditElementSetNumbersGroup::~vtkKWMimxEditElementSetNumbersGroup()
{
  if(this->ObjectListComboBox)
     this->ObjectListComboBox->Delete();
  if(this->ElementSetComboBox)
          this->ElementSetComboBox->Delete();
  if(this->StartingElementNumberEntry)
          this->StartingElementNumberEntry->Delete();
}
//----------------------------------------------------------------------------
void vtkKWMimxEditElementSetNumbersGroup::CreateWidget()
{
  if(this->IsCreated())
  {
    vtkErrorMacro("class already created");
    return;
  }

  this->Superclass::CreateWidget();
  if(!this->ObjectListComboBox) 
  {
     this->ObjectListComboBox = vtkKWComboBoxWithLabel::New();
  }
  this->MainFrame->SetParent(this->GetParent());
  this->MainFrame->Create();
  this->MainFrame->SetLabelText("Edit Element Numbers of a given element set");

  this->GetApplication()->Script(
    "pack %s -side top -anchor nw -expand n -padx 2 -pady 0 -fill x", 
    this->MainFrame->GetWidgetName());

  ObjectListComboBox->SetParent(this->MainFrame->GetFrame());
  ObjectListComboBox->Create();
  ObjectListComboBox->SetLabelText("Mesh : ");
  ObjectListComboBox->SetLabelWidth(15);
  ObjectListComboBox->GetWidget()->ReadOnlyOn();
  ObjectListComboBox->GetWidget()->SetCommand(this, "SelectionChangedCallback");
  this->GetApplication()->Script(
    "pack %s -side top -anchor nw -expand y -padx 2 -pady 6 -fill x", 
    ObjectListComboBox->GetWidgetName());

  // for element set listing
  if(!this->ElementSetComboBox) 
  {
          this->ElementSetComboBox = vtkKWComboBoxWithLabel::New();
  }
  ElementSetComboBox->SetParent(this->MainFrame->GetFrame());
  ElementSetComboBox->Create();
  ElementSetComboBox->SetLabelText("Element Set : ");
  ElementSetComboBox->SetLabelWidth(15);
  ElementSetComboBox->GetWidget()->ReadOnlyOn();
  this->ElementSetComboBox->GetWidget()->SetCommand(this, "ElementSetChangedCallback");
  this->GetApplication()->Script(
          "pack %s -side top -anchor nw -expand y -padx 2 -pady 6 -fill x", 
          ElementSetComboBox->GetWidgetName());

  //Young's modulus
  if (!this->StartingElementNumberEntry)
          this->StartingElementNumberEntry = vtkKWEntryWithLabel::New();

  this->StartingElementNumberEntry->SetParent(this->MainFrame->GetFrame());
  this->StartingElementNumberEntry->Create();
  //this->StartingElementNumberEntry->SetWidth(10);
  this->StartingElementNumberEntry->SetLabelWidth(15);
  this->StartingElementNumberEntry->SetLabelText("Start Element Num : ");
//  this->StartingElementNumberEntry->GetWidget()->SetCommand(this, "RadiusChangeCallback");
  this->StartingElementNumberEntry->GetWidget()->SetRestrictValueToInteger();
  this->StartingElementNumberEntry->GetWidget()->SetValueAsInt(1);
  this->GetApplication()->Script(
          "pack %s -side top -anchor nw -expand y -padx 2 -pady 6 -fill x", 
          this->StartingElementNumberEntry->GetWidgetName());

  this->ApplyButton->SetParent(this->MainFrame->GetFrame());
  this->ApplyButton->Create();
  this->ApplyButton->SetText("Apply");
  this->ApplyButton->SetCommand(this, "EditElementSetNumbersApplyCallback");
  this->GetApplication()->Script(
          "pack %s -side left -anchor nw -expand y -padx 20 -pady 6", 
          this->ApplyButton->GetWidgetName());
/*
  this->DoneButton->SetParent(this->MainFrame->GetFrame());
  this->DoneButton->Create();
  this->DoneButton->SetText("Done");
  this->DoneButton->SetCommand(this, "EditElementSetNumbersDoneCallback");
  this->GetApplication()->Script(
    "pack %s -side left -anchor nw -expand y -padx 20 -pady 6", 
    this->DoneButton->GetWidgetName());
*/
  this->CancelButton->SetParent(this->MainFrame->GetFrame());
  this->CancelButton->Create();
  this->CancelButton->SetText("Cancel");
  this->CancelButton->SetCommand(this, "EditElementSetNumbersCancelCallback");
  this->GetApplication()->Script(
    "pack %s -side right -anchor ne -expand y -padx 20 -pady 6", 
    this->CancelButton->GetWidgetName());

}
//----------------------------------------------------------------------------
void vtkKWMimxEditElementSetNumbersGroup::Update()
{
        this->UpdateEnableState();
}
//---------------------------------------------------------------------------
void vtkKWMimxEditElementSetNumbersGroup::UpdateEnableState()
{
        this->UpdateObjectLists();
        this->Superclass::UpdateEnableState();
}
//----------------------------------------------------------------------------
int vtkKWMimxEditElementSetNumbersGroup::EditElementSetNumbersApplyCallback()
{
        vtkMimxErrorCallback *callback = this->GetMimxMainWindow()->GetErrorCallback();
        if(!strcmp(this->ObjectListComboBox->GetWidget()->GetValue(),""))
        {
                callback->ErrorMessage("FE Mesh selection required");
                return 0;
        }

  vtkKWComboBox *combobox = this->ObjectListComboBox->GetWidget();
    const char *name = combobox->GetValue();

        int num = combobox->GetValueIndex(name);
        if(num < 0 || num > combobox->GetNumberOfValues()-1)
        {
                callback->ErrorMessage("Choose valid FE Mesh");
                combobox->SetValue("");
                return 0;
        }
        
        int startelenum = this->StartingElementNumberEntry->GetWidget()->GetValueAsInt();

        if(startelenum < 1)
        {
                callback->ErrorMessage("Starting element number cannot be < 1");
                return 0;
        }

        vtkMimxMeshActor *meshActor = vtkMimxMeshActor::SafeDownCast(
                this->FEMeshList->GetItem(combobox->GetValueIndex(name)));
  vtkUnstructuredGrid *ugrid = meshActor->GetDataSet();
        
  const char *elementsetname = this->ElementSetComboBox->GetWidget()->GetValue();

  if(!strcmp(elementsetname,""))
  {
          callback->ErrorMessage("Choose valid element set name");
          return 0;
  }

  meshActor->ChangeElementSetNumbers(elementsetname, startelenum);  
  return 1;
}
//----------------------------------------------------------------------------
void vtkKWMimxEditElementSetNumbersGroup::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
//----------------------------------------------------------------------------
void vtkKWMimxEditElementSetNumbersGroup::EditElementSetNumbersCancelCallback()
{
//  this->MainFrame->UnpackChildren();
  this->GetApplication()->Script("pack forget %s", this->MainFrame->GetWidgetName());
  this->MenuGroup->SetMenuButtonsEnabled(1);
    this->GetMimxMainWindow()->GetMainUserInterfacePanel()->GetMimxMainNotebook()->SetEnabled(1);
}
//-----------------------------------------------------------------------------
void vtkKWMimxEditElementSetNumbersGroup::UpdateObjectLists()
{
        this->ObjectListComboBox->GetWidget()->DeleteAllValues();
        
        int defaultItem = -1;
        for (int i = 0; i < this->FEMeshList->GetNumberOfItems(); i++)
        {
                ObjectListComboBox->GetWidget()->AddValue(
                        this->FEMeshList->GetItem(i)->GetFileName());
          int viewedItem = this->GetMimxMainWindow()->GetRenderWidget()->GetRenderer()->HasViewProp(
                                         this->FEMeshList->GetItem(i)->GetActor());
    if ( (defaultItem == -1) && ( viewedItem ) )
                {
                  defaultItem = i;
                }                               
        }
        if (defaultItem != -1)
  {
    ObjectListComboBox->GetWidget()->SetValue(
          this->FEMeshList->GetItem(defaultItem)->GetFileName());
  }
        this->SelectionChangedCallback(ObjectListComboBox->GetWidget()->GetValue());
}
//--------------------------------------------------------------------------------
void vtkKWMimxEditElementSetNumbersGroup::SelectionChangedCallback(const char *Selection)
{
        if(!strcmp(Selection,""))
        {
                return;
        }
        vtkKWComboBox *combobox = this->ObjectListComboBox->GetWidget();
        vtkUnstructuredGrid *ugrid = vtkMimxMeshActor::SafeDownCast(
                this->FEMeshList->GetItem(combobox->GetValueIndex(Selection)))->GetDataSet();

        // populate the element set list
        this->ElementSetComboBox->GetWidget()->DeleteAllValues();
        int i;
        vtkStringArray *strarray = vtkStringArray::SafeDownCast(
                ugrid->GetFieldData()->GetAbstractArray("Element_Set_Names"));

        int numarrrays = strarray->GetNumberOfValues();

        for (i=0; i<numarrrays; i++)
        {
                this->ElementSetComboBox->GetWidget()->AddValue(
                        strarray->GetValue(i));
        }
        this->ElementSetComboBox->GetWidget()->SetValue( strarray->GetValue(0) );
}
//-------------------------------------------------------------------------------------
void vtkKWMimxEditElementSetNumbersGroup::ElementSetChangedCallback(const char *Selection)
{
        //if(!strcmp(Selection,""))
        //{
        //      return;
        //}
        //// get the femesh
        //vtkKWComboBox *combobox = this->ObjectListComboBox->GetWidget();
        //vtkUnstructuredGrid *ugrid = vtkMimxMeshActor::SafeDownCast(
        //      this->FEMeshList->GetItem(combobox->GetValueIndex(
        //      ObjectListComboBox->GetWidget()->GetValue())))->GetDataSet();

        //char young[256];
        //strcpy(young, Selection);
        //strcat(young, "_Constant_Youngs_Modulus");

        //char poisson[256];
        //strcpy(poisson, Selection);
        //strcat(poisson, "_Constant_Poissons_Ratio");

        //// check if the field data exists.
        //vtkFloatArray *Earray = vtkFloatArray::SafeDownCast(
        //      ugrid->GetFieldData()->GetArray(young));        

        //vtkFloatArray *Nuarray = vtkFloatArray::SafeDownCast(
        //      ugrid->GetFieldData()->GetArray(poisson));      

        //float startelenum = -1.0;
        //float poissonsratio = -1.0;
        //if(Earray)    startelenum = Earray->GetValue(0);
        //if(Nuarray)   poissonsratio = Nuarray->GetValue(0);
        //// youngs modulus and poissons ratio data exists
        //this->StartingElementNumberEntry->GetWidget()->SetValueAsDouble(startelenum);
        //this->PoissonsRatioEntry->GetWidget()->SetValueAsDouble(poissonsratio);
}
//----------------------------------------------------------------------------------------
