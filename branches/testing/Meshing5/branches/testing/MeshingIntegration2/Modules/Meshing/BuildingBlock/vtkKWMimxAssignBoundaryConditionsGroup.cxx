/*=========================================================================

Program:   MIMX Meshing Toolkit
Module:    $RCSfile: vtkKWMimxAssignBoundaryConditionsGroup.cxx,v $
Language:  C++
Date:      $Date: 2008/05/05 19:30:08 $
Version:   $Revision: 1.7 $

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

#include "vtkKWMimxAssignBoundaryConditionsGroup.h"
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
#include "vtkKWCheckButtonWithLabel.h"
#include "vtkKWCheckButton.h"
#include "vtkSphereSource.h"

#include "vtkRenderer.h"
#include "vtkPolyData.h"
#include "vtkPolyDataMapper.h"
#include "vtkGlyph3D.h"
#include "vtkPointSet.h"
#include "vtkProperty.h"

#include <vtksys/stl/list>
#include <vtksys/stl/string>
#include <vtksys/stl/algorithm>

// define the option types
#define VTK_KW_OPTION_NONE         0
#define VTK_KW_OPTION_LOAD                 1

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkKWMimxAssignBoundaryConditionsGroup);
vtkCxxRevisionMacro(vtkKWMimxAssignBoundaryConditionsGroup, "$Revision: 1.7 $");

//----------------------------------------------------------------------------
vtkKWMimxAssignBoundaryConditionsGroup::vtkKWMimxAssignBoundaryConditionsGroup()
{
  this->ObjectListComboBox = NULL;
  this->NodeSetComboBox = NULL;
  this->BoundaryConditionTypeComboBox = NULL;
  this->StepFrame = NULL;
  this->StepNumberComboBox = NULL;
  this->AddStepPushButton = NULL;
  this->DirectionFrame = NULL;
  this->DirectionXEntry = NULL;
  this->DirectionYEntry = NULL;
  this->DirectionZEntry = NULL;
  this->ViewBoundaryConditionsButton = NULL;
  this->CancelStatus = 0;
  this->GlyphActor = NULL;
  this->ViewFrame = NULL;
  this->ViewDirectionComboBox = NULL;
}

//----------------------------------------------------------------------------
vtkKWMimxAssignBoundaryConditionsGroup::~vtkKWMimxAssignBoundaryConditionsGroup()
{
  if(this->ObjectListComboBox)
     this->ObjectListComboBox->Delete();
  if(this->NodeSetComboBox)
          this->NodeSetComboBox->Delete();
  if(this->BoundaryConditionTypeComboBox)
          this->BoundaryConditionTypeComboBox->Delete();
  if(this->StepFrame)
          this->StepFrame->Delete();
  if(this->StepNumberComboBox)
          this->StepNumberComboBox->Delete();
  if(this->AddStepPushButton)
          this->AddStepPushButton->Delete();
  if(this->DirectionFrame)
          this->DirectionFrame->Delete();
  if(this->DirectionXEntry)
          this->DirectionXEntry->Delete();
  if(this->DirectionYEntry)
          this->DirectionYEntry->Delete();
  if(this->DirectionZEntry)
          this->DirectionZEntry->Delete();  
  if(this->ViewBoundaryConditionsButton)
          this->ViewBoundaryConditionsButton->Delete();
  if(this->GlyphActor)
          this->GlyphActor->Delete();
  if(this->ViewFrame)
          this->ViewFrame->Delete();
  if(this->ViewDirectionComboBox)
          this->ViewDirectionComboBox->Delete();
}
//----------------------------------------------------------------------------
void vtkKWMimxAssignBoundaryConditionsGroup::CreateWidget()
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
  this->MainFrame->SetLabelText("Boundary Conditions");

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

  //step details
 if(!this->StepFrame)
         this->StepFrame = vtkKWFrameWithLabel::New();

 this->StepFrame->SetParent(this->MainFrame->GetFrame());
 this->StepFrame->Create();
 this->StepFrame->SetLabelText("Step Details");
 this->GetApplication()->Script(
         "pack %s -side top -anchor nw -expand y -padx 2 -pady 6 -fill x", 
         this->StepFrame->GetWidgetName());

 if(!this->StepNumberComboBox)  
         this->StepNumberComboBox = vtkKWComboBoxWithLabel::New();
 this->StepNumberComboBox->SetParent(this->StepFrame->GetFrame());
 this->StepNumberComboBox->Create();
 this->StepNumberComboBox->SetLabelText("Step Number : ");
 this->StepNumberComboBox->GetWidget()->ReadOnlyOn();
 this->StepNumberComboBox->GetWidget()->SetCommand(this, "StepNumberChangedCallback");
 this->GetApplication()->Script(
         "pack %s -side left -anchor nw -expand y -padx 2 -pady 2",  
         this->StepNumberComboBox->GetWidgetName());

 if(!this->AddStepPushButton)   this->AddStepPushButton = vtkKWPushButton::New();
 this->AddStepPushButton->SetParent(this->StepFrame->GetFrame());
 this->AddStepPushButton->Create();
 this->AddStepPushButton->SetText("Add Step Number");
 this->AddStepPushButton->SetCommand(this, "AddStepNumberCallback");
 this->GetApplication()->Script(
          "pack %s -side top -anchor nw -expand y -padx 2 -pady 2",
         this->AddStepPushButton->GetWidgetName());

  // for Node Set listing
  if(!this->NodeSetComboBox)    
  {
          this->NodeSetComboBox = vtkKWComboBoxWithLabel::New();
  }
  this->NodeSetComboBox->SetParent(this->MainFrame->GetFrame());
  this-> NodeSetComboBox->Create();
  this->NodeSetComboBox->SetLabelText("Node Set : ");
  this->NodeSetComboBox->GetWidget()->ReadOnlyOn();
  this->NodeSetComboBox->GetWidget()->SetCommand(this, "NodeSetChangedCallback");
  this->GetApplication()->Script(
          "pack %s -side top -anchor nw -expand y -padx 2 -pady 6 -fill x", 
          NodeSetComboBox->GetWidgetName());
//
  if(!this->BoundaryConditionTypeComboBox)      
  {
          this->BoundaryConditionTypeComboBox = vtkKWComboBoxWithLabel::New();
  }
  this->BoundaryConditionTypeComboBox->SetParent(this->MainFrame->GetFrame());
  this->BoundaryConditionTypeComboBox->Create();
  this->BoundaryConditionTypeComboBox->SetLabelText("Type : ");
  this->BoundaryConditionTypeComboBox->SetLabelWidth(15);
  this->BoundaryConditionTypeComboBox->GetWidget()->AddValue("Force");
  this->BoundaryConditionTypeComboBox->GetWidget()->AddValue("Displacement");
  this->BoundaryConditionTypeComboBox->GetWidget()->AddValue("Rotation");
  this->BoundaryConditionTypeComboBox->GetWidget()->AddValue("Moment");
  this->BoundaryConditionTypeComboBox->GetWidget()->ReadOnlyOn();
  this->BoundaryConditionTypeComboBox->GetWidget()->SetCommand(
          this, "BoundaryConditionTypeSelectionChangedCallback");
  this->GetApplication()->Script(
          "pack %s -side top -anchor nw -expand y -padx 2 -pady 6 -fill x", 
          this->BoundaryConditionTypeComboBox->GetWidgetName());

        //direction details
  if(!this->DirectionFrame)
          this->DirectionFrame = vtkKWFrameWithLabel::New();

  this->DirectionFrame->SetParent(this->MainFrame->GetFrame());
  this->DirectionFrame->Create();
  this->DirectionFrame->SetLabelText("");
  this->GetApplication()->Script(
          "pack %s -side top -anchor nw -expand y -padx 2 -pady 6 -fill x", 
          this->DirectionFrame->GetWidgetName());

  //direction choice X
  if (!this->DirectionXEntry)
          this->DirectionXEntry = vtkKWEntryWithLabel::New();

  this->DirectionXEntry->SetParent(this->DirectionFrame->GetFrame());
  this->DirectionXEntry->Create();
  this->DirectionXEntry->SetLabelText("X : ");
  this->DirectionXEntry->GetWidget()->SetRestrictValueToDouble();

  this->GetApplication()->Script(
          "pack %s -side left -anchor nw -expand y -padx 6 -pady 2", 
          this->DirectionXEntry->GetWidgetName());

  //Y
  if (!this->DirectionYEntry)
          this->DirectionYEntry = vtkKWEntryWithLabel::New();

  this->DirectionYEntry->SetParent(this->DirectionFrame->GetFrame());
  this->DirectionYEntry->Create();
  this->DirectionYEntry->SetLabelText("Y : ");
  this->DirectionYEntry->GetWidget()->SetRestrictValueToDouble();

  this->GetApplication()->Script(
          "pack %s -side left -anchor nw -expand y -padx 6 -pady 2", 
          this->DirectionYEntry->GetWidgetName());
  //Z
  if (!this->DirectionZEntry)
          this->DirectionZEntry = vtkKWEntryWithLabel::New();

  this->DirectionZEntry->SetParent(this->DirectionFrame->GetFrame());
  this->DirectionZEntry->Create();
  this->DirectionZEntry->SetLabelText("Z : ");
  this->DirectionZEntry->GetWidget()->SetRestrictValueToDouble();

  this->GetApplication()->Script(
          "pack %s -side left -anchor nw -expand y -padx 6 -pady 2", 
          this->DirectionZEntry->GetWidgetName());

  if (!this->ViewFrame)
          this->ViewFrame = vtkKWFrameWithLabel::New();
  this->ViewFrame->SetParent( this->MainFrame->GetFrame() );
  this->ViewFrame->Create();
  this->ViewFrame->SetLabelText("View");
  this->GetApplication()->Script("pack %s -side top -anchor nw -expand n -fill x -padx 2 -pady 2",
          this->ViewFrame->GetWidgetName() );    
  this->ViewFrame->CollapseFrame();

  if (!this->ViewBoundaryConditionsButton)
          this->ViewBoundaryConditionsButton = vtkKWCheckButtonWithLabel::New();
  this->ViewBoundaryConditionsButton->SetParent(this->ViewFrame->GetFrame());
  this->ViewBoundaryConditionsButton->Create();
  this->ViewBoundaryConditionsButton->GetWidget()->SetCommand(this, "ViewBoundaryConditionsCallback");
  this->ViewBoundaryConditionsButton->GetWidget()->SetText("View Boundary Conditions");
  //  this->ViewPropertyButton->GetWidget()->SetEnabled( 0 );
  this->GetApplication()->Script(
          "pack %s -side top -anchor nw -expand n -fill x -padx 2 -pady 2", 
          this->ViewBoundaryConditionsButton->GetWidgetName());

  if(!this->ViewDirectionComboBox)      
          this->ViewDirectionComboBox = vtkKWComboBoxWithLabel::New();
  this->ViewDirectionComboBox->SetParent(this->ViewFrame->GetFrame());
  this->ViewDirectionComboBox->Create();
  this->ViewDirectionComboBox->SetLabelText("Direction : ");
  this->ViewDirectionComboBox->GetWidget()->ReadOnlyOn();
  this->ViewDirectionComboBox->GetWidget()->AddValue("X");
  this->ViewDirectionComboBox->GetWidget()->AddValue("Y");
  this->ViewDirectionComboBox->GetWidget()->AddValue("Z");
//  this->ViewDirectionComboBox->GetWidget()->SetCommand(this, "StepNumberChangedCallback");
  this->GetApplication()->Script(
          "pack %s -side left -anchor nw -expand y -padx 2 -pady 2",  
          this->ViewDirectionComboBox->GetWidgetName());

  this->ApplyButton->SetParent(this->MainFrame->GetFrame());
  this->ApplyButton->Create();
  this->ApplyButton->SetText("Apply");
  this->ApplyButton->SetCommand(this, "AssignBoundaryConditionsApplyCallback");
  this->GetApplication()->Script(
          "pack %s -side left -anchor nw -expand y -padx 20 -pady 6", 
          this->ApplyButton->GetWidgetName());
/*
  this->DoneButton->SetParent(this->MainFrame->GetFrame());
  this->DoneButton->Create();
  this->DoneButton->SetText("Done");
  this->DoneButton->SetCommand(this, "AssignBoundaryConditionsDoneCallback");
  this->GetApplication()->Script(
    "pack %s -side left -anchor nw -expand y -padx 20 -pady 6", 
    this->DoneButton->GetWidgetName());
*/
  this->CancelButton->SetParent(this->MainFrame->GetFrame());
  this->CancelButton->Create();
  this->CancelButton->SetText("Cancel");
  this->CancelButton->SetCommand(this, "AssignBoundaryConditionsCancelCallback");
  this->GetApplication()->Script(
    "pack %s -side right -anchor ne -expand y -padx 20 -pady 6", 
    this->CancelButton->GetWidgetName());

}
//----------------------------------------------------------------------------
void vtkKWMimxAssignBoundaryConditionsGroup::Update()
{
        this->UpdateEnableState();
}
//---------------------------------------------------------------------------
void vtkKWMimxAssignBoundaryConditionsGroup::UpdateEnableState()
{
        this->UpdateObjectLists();
        this->Superclass::UpdateEnableState();
}
//----------------------------------------------------------------------------
int vtkKWMimxAssignBoundaryConditionsGroup::AssignBoundaryConditionsApplyCallback()
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
        
  vtkUnstructuredGrid *ugrid = vtkMimxMeshActor::SafeDownCast(
          this->FEMeshList->GetItem(combobox->GetValueIndex(name)))->GetDataSet();
        
  const char *NodeSetname = this->NodeSetComboBox->GetWidget()->GetValue();

  if(!strcmp(NodeSetname,""))
  {
          callback->ErrorMessage("Choose valid Node Set name");
          return 0;
  }

  const char *boundaryconditiontype = 
          this->BoundaryConditionTypeComboBox->GetWidget()->GetValue();
  const char *entryx = this->DirectionXEntry->GetWidget()->GetValue();
  const char *entryy = this->DirectionYEntry->GetWidget()->GetValue();
  const char *entryz = this->DirectionZEntry->GetWidget()->GetValue();

  if(!strcmp(boundaryconditiontype,""))
  {
          callback->ErrorMessage("Choose Boundary condition type");
          return 0;
  }

  if(!strcmp(entryx,"") && !strcmp(entryy,"") && !strcmp(entryz,""))
  {
          callback->ErrorMessage("Magnitude values are empty");
          return 0;
  }

  const char *stepnum = this->StepNumberComboBox->GetWidget()->GetValue();
  if(strcmp(entryx,""))
  {
          char boundname[256];

          this->ConcatenateStrings("Step", stepnum, NodeSetname, 
                  boundaryconditiontype, "X", boundname);

          vtkFloatArray *boundarray = vtkFloatArray::SafeDownCast(
                  ugrid->GetFieldData()->GetArray(boundname));
           if(boundarray)
          {
                  ugrid->GetFieldData()->RemoveArray(boundname);
          }
          // create a new field data entry
           boundarray = vtkFloatArray::New();
           boundarray->SetName(boundname);
           double entry = this->DirectionXEntry->GetWidget()->GetValueAsDouble();
           boundarray->InsertNextValue(entry);
           ugrid->GetFieldData()->AddArray(boundarray);
           boundarray->Delete();
           // update the number of steps in the boundary condition
           int intstepnum = this->StepNumberComboBox->GetWidget()->GetValueAsInt();
           vtkIntArray *BoundCond = vtkIntArray::SafeDownCast(
                   ugrid->GetFieldData()->GetArray("Boundary_Condition_Number_Of_Steps"));

           if(intstepnum == BoundCond->GetValue(0)+1)
           {
                   BoundCond->SetValue(0, intstepnum);
           }
  }

  if(strcmp(entryy,""))
  {
          char boundname[256];

          this->ConcatenateStrings("Step", stepnum, NodeSetname, 
                  boundaryconditiontype, "Y", boundname);

          vtkFloatArray *boundarray = vtkFloatArray::SafeDownCast(
                  ugrid->GetFieldData()->GetArray(boundname));
          if(boundarray)
          {
                  ugrid->GetFieldData()->RemoveArray(boundname);
          }
          // create a new field data entry
          boundarray = vtkFloatArray::New();
          boundarray->SetName(boundname);
          double entry = this->DirectionYEntry->GetWidget()->GetValueAsDouble();
          boundarray->InsertNextValue(entry);
          ugrid->GetFieldData()->AddArray(boundarray);
          boundarray->Delete();
          // update the number of steps in the boundary condition
          int intstepnum = this->StepNumberComboBox->GetWidget()->GetValueAsInt();
          vtkIntArray *BoundCond = vtkIntArray::SafeDownCast(
                  ugrid->GetFieldData()->GetArray("Boundary_Condition_Number_Of_Steps"));

          if(intstepnum == BoundCond->GetValue(0)+1)
          {
                  BoundCond->SetValue(0, intstepnum);
          }
  }

  if(strcmp(entryz,""))
  {
          char boundname[256];

          this->ConcatenateStrings("Step", stepnum, NodeSetname, 
                  boundaryconditiontype, "Z", boundname);

          vtkFloatArray *boundarray = vtkFloatArray::SafeDownCast(
                  ugrid->GetFieldData()->GetArray(boundname));
          if(boundarray)
          {
                  ugrid->GetFieldData()->RemoveArray(boundname);
          }
          // create a new field data entry
          boundarray = vtkFloatArray::New();
          boundarray->SetName(boundname);
          double entry = this->DirectionZEntry->GetWidget()->GetValueAsDouble();
          boundarray->InsertNextValue(entry);
          ugrid->GetFieldData()->AddArray(boundarray);
          boundarray->Delete();
          // update the number of steps in the boundary condition
          int intstepnum = this->StepNumberComboBox->GetWidget()->GetValueAsInt();
          vtkIntArray *BoundCond = vtkIntArray::SafeDownCast(
                  ugrid->GetFieldData()->GetArray("Boundary_Condition_Number_Of_Steps"));

          if(intstepnum == BoundCond->GetValue(0)+1)
          {
                  BoundCond->SetValue(0, intstepnum);
          }
  }

  return 1;
}
//----------------------------------------------------------------------------
void vtkKWMimxAssignBoundaryConditionsGroup::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
//----------------------------------------------------------------------------
void vtkKWMimxAssignBoundaryConditionsGroup::AssignBoundaryConditionsCancelCallback()
{
//  this->MainFrame->UnpackChildren();
        this->CancelStatus = 1;
  this->GetApplication()->Script("pack forget %s", this->MainFrame->GetWidgetName());
  this->MenuGroup->SetMenuButtonsEnabled(1);
    this->GetMimxMainWindow()->GetMainUserInterfacePanel()->GetMimxMainNotebook()->SetEnabled(1);
        this->CancelStatus = 0;
}
//-----------------------------------------------------------------------------
void vtkKWMimxAssignBoundaryConditionsGroup::UpdateObjectLists()
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
void vtkKWMimxAssignBoundaryConditionsGroup::SelectionChangedCallback(const char *Selection)
{
        if(this->CancelStatus)  return;
        int i;

        if(!strcmp(Selection,""))
        {
                return;
        }
        vtkKWComboBox *combobox = this->ObjectListComboBox->GetWidget();
        vtkUnstructuredGrid *ugrid = vtkMimxMeshActor::SafeDownCast(
                this->FEMeshList->GetItem(combobox->GetValueIndex(Selection)))->GetDataSet();
        //      
        vtkIntArray *BoundCond = vtkIntArray::SafeDownCast(
                ugrid->GetFieldData()->GetArray("Boundary_Condition_Number_Of_Steps"));
        int NumberOfSteps;
        if(!BoundCond)
        {
                BoundCond = vtkIntArray::New();
                BoundCond->SetNumberOfValues(1);
                BoundCond->SetValue(0,1);
                BoundCond->SetName("Boundary_Condition_Number_Of_Steps");
                ugrid->GetFieldData()->AddArray(BoundCond);
                BoundCond->Delete();
                NumberOfSteps = 0;
        }
        else
        {
                NumberOfSteps = BoundCond->GetValue(0);
        }
        this->StepNumberComboBox->GetWidget()->DeleteAllValues();
        
        if(NumberOfSteps == 0)  NumberOfSteps = 1;
        char num[10];
        for (i=0; i<NumberOfSteps; i++)
        {
                sprintf(num, "%d", i+1);
                this->StepNumberComboBox->GetWidget()->AddValue(num);
        }
        this->StepNumberComboBox->GetWidget()->SetValue(num);   
        // populate the Node Set list
        this->NodeSetComboBox->GetWidget()->DeleteAllValues();
        vtkStringArray *strarray = vtkStringArray::SafeDownCast(
                ugrid->GetFieldData()->GetAbstractArray("Node_Set_Names"));

        int numarrrays = strarray->GetNumberOfValues();

        for (i=0; i<numarrrays; i++)
        {
                this->NodeSetComboBox->GetWidget()->AddValue(
                        strarray->GetValue(i));
        }
        this->NodeSetComboBox->GetWidget()->SetValue(strarray->GetValue(i-1));
    this->NodeSetChangedCallback(strarray->GetValue(i-1));
}
//-------------------------------------------------------------------------------------
void vtkKWMimxAssignBoundaryConditionsGroup::NodeSetChangedCallback(const char *Selection)
{
        if(this->CancelStatus)  return;

        if(!strcmp(Selection,""))
        {
                return;
        }
        if(strcmp(this->BoundaryConditionTypeComboBox->GetWidget()->GetValue(), ""))
        {
                this->BoundaryConditionTypeComboBox->GetWidget()->SetValue(
                        this->BoundaryConditionTypeComboBox->GetWidget()->GetValue());
        }
        else
        {
                this->BoundaryConditionTypeComboBox->GetWidget()->SetValue("Force");
        }
        this->BoundaryConditionTypeSelectionChangedCallback(
                this->BoundaryConditionTypeComboBox->GetWidget()->GetValue());
}
//----------------------------------------------------------------------------------------
void vtkKWMimxAssignBoundaryConditionsGroup::
        BoundaryConditionTypeSelectionChangedCallback(const char *Selection)
{
        if(this->CancelStatus)  return;
        if(!strcmp(Selection,""))
        {
                return;
        }
        if(!strcmp("Force", Selection) || !strcmp("Displacement", Selection))
        {
                this->DirectionFrame->SetLabelText("Along : ");
        }
        else
        {
                this->DirectionFrame->SetLabelText("About : ");
        }
        this->GetValue();
}
//----------------------------------------------------------------------------------------
//void vtkKWMimxAssignBoundaryConditionsGroup::
//      DirectionSelectionCallback(const char *Selection)
//{
//      if(!strcmp(this->NodeSetComboBox->GetWidget()->GetValue(),""))
//      {
//              return;
//      }
//      if(!strcmp(Selection,""))
//      {
//              return;
//      }
//      if(!strcmp(this->BoundaryConditionTypeComboBox->GetWidget()->GetValue(),""))
//      {
//              return;
//      }
//      this->GetValue();
//}
//----------------------------------------------------------------------------------------
void vtkKWMimxAssignBoundaryConditionsGroup::GetValue()
{
        // list all the three values.
        if(!strcmp(this->ObjectListComboBox->GetWidget()->GetValue(),""))
        {
                return;
        }

        if(!strcmp(this->NodeSetComboBox->GetWidget()->GetValue(),""))
        {
                return;
        }
        
        vtkUnstructuredGrid *ugrid = vtkMimxMeshActor::SafeDownCast(
                this->FEMeshList->GetItem(this->ObjectListComboBox->GetWidget()->GetValueIndex(
                ObjectListComboBox->GetWidget()->GetValue())))->GetDataSet();

        // stored value for direction
        // X
        char name[256];
        strcpy(name, "Step_");
        strcat(name, this->StepNumberComboBox->GetWidget()->GetValue());
        strcat(name,"_");
        strcat(name, this->NodeSetComboBox->GetWidget()->GetValue());
        strcat(name, "_");
        strcat(name, this->BoundaryConditionTypeComboBox->GetWidget()->GetValue());
        strcat(name,"_");
        strcat(name, "X");

        vtkDataArray *dataarray = ugrid->GetFieldData()->GetArray(name);
        float Value = 0.0;
        if(dataarray)
        {
                Value = vtkFloatArray::SafeDownCast(dataarray)->GetValue(0);
                this->DirectionXEntry->GetWidget()->SetValueAsDouble(Value);
        }
        else{
                this->DirectionXEntry->GetWidget()->SetValue("");
        }

        // Y
        name[256];
        strcpy(name, "Step_");
        strcat(name, this->StepNumberComboBox->GetWidget()->GetValue());
        strcat(name,"_");
        strcat(name, this->NodeSetComboBox->GetWidget()->GetValue());
        strcat(name, "_");
        strcat(name, this->BoundaryConditionTypeComboBox->GetWidget()->GetValue());
        strcat(name,"_");
        strcat(name, "Y");

        dataarray = ugrid->GetFieldData()->GetArray(name);
        Value = 0.0;
        if(dataarray)
        {
                Value = vtkFloatArray::SafeDownCast(dataarray)->GetValue(0);
                this->DirectionYEntry->GetWidget()->SetValueAsDouble(Value);
        }
        else{
                this->DirectionYEntry->GetWidget()->SetValue("");
        }

        // Z
        name[256];
        strcpy(name, "Step_");
        strcat(name, this->StepNumberComboBox->GetWidget()->GetValue());
        strcat(name,"_");
        strcat(name, this->NodeSetComboBox->GetWidget()->GetValue());
        strcat(name, "_");
        strcat(name, this->BoundaryConditionTypeComboBox->GetWidget()->GetValue());
        strcat(name,"_");
        strcat(name, "Z");

        dataarray = ugrid->GetFieldData()->GetArray(name);
        Value = 0.0;
        if(dataarray)
        {
                Value = vtkFloatArray::SafeDownCast(dataarray)->GetValue(0);
                this->DirectionZEntry->GetWidget()->SetValueAsDouble(Value);
        }
        else{
                this->DirectionZEntry->GetWidget()->SetValue("");
        }
}
//-----------------------------------------------------------------------------------------
void vtkKWMimxAssignBoundaryConditionsGroup::AddStepNumberCallback()
{
        if(this->CancelStatus)  return;
        vtkMimxErrorCallback *callback = this->GetMimxMainWindow()->GetErrorCallback();
        // check if the given step number has any boundary conditions
        if(!strcmp(this->ObjectListComboBox->GetWidget()->GetValue(),""))
        {
                callback->ErrorMessage("FE Mesh selection required");
                return;
        }

        vtkKWComboBox *combobox = this->ObjectListComboBox->GetWidget();
        const char *name = combobox->GetValue();

        int num = combobox->GetValueIndex(name);
        if(num < 0 || num > combobox->GetNumberOfValues()-1)
        {
                callback->ErrorMessage("Choose valid FE Mesh");
                combobox->SetValue("");
                return;
        }

        vtkUnstructuredGrid *ugrid = vtkMimxMeshActor::SafeDownCast(
                this->FEMeshList->GetItem(combobox->GetValueIndex(name)))->GetDataSet();

        if(!strcmp(this->NodeSetComboBox->GetWidget()->GetValue(),""))
        {
                callback->ErrorMessage("Select node set");
                return;
        }
        //
        if(this->IsStepEmpty(ugrid))
        {
                callback->ErrorMessage("Cannot add a new step because the final step is empty");
                return;
        }
        char numentries[10];
        sprintf(numentries, "%d", this->StepNumberComboBox->GetWidget()->GetNumberOfValues()+1);
        this->StepNumberComboBox->GetWidget()->AddValue(numentries);
        this->StepNumberComboBox->GetWidget()->SetValue(numentries);
}
//-----------------------------------------------------------------------------------------
void vtkKWMimxAssignBoundaryConditionsGroup::StepNumberChangedCallback(
        const char *StepNum)
{
        if(this->CancelStatus)  return;
        this->GetValue();       
}
//-----------------------------------------------------------------------------------------
void vtkKWMimxAssignBoundaryConditionsGroup::ConcatenateStrings(
        const char* Step, const char* Num, const char* NodeSetName, 
        const char* Type, const char* Direction, char *Name)
{
        strcpy(Name, Step);
        strcat(Name, "_");
        strcat(Name,Num);
        strcat(Name, "_");
        strcat(Name, NodeSetName);
        strcat(Name, "_");
        strcat(Name,Type);
        strcat(Name, "_");
        strcat(Name, Direction);
}
//------------------------------------------------------------------------------------------
int vtkKWMimxAssignBoundaryConditionsGroup::IsStepEmpty(vtkUnstructuredGrid *ugrid)
{
        //
        char Concatenate[256];
        int i;
        vtkStringArray *nodesetnamestring = vtkStringArray::SafeDownCast(
                ugrid->GetFieldData()->GetAbstractArray("Node_Set_Names"));
        int numsteps = this->StepNumberComboBox->GetWidget()->GetNumberOfValues();
        char stepnum[10];
        sprintf(stepnum, "%d", numsteps);

        if(!nodesetnamestring)  return 1;
        for(i=0; i<nodesetnamestring->GetNumberOfValues(); i++)
        {
                Concatenate[256];
                const char* nodesetname =  nodesetnamestring->GetValue(i);
                this->ConcatenateStrings("Step", stepnum, nodesetname, "Force", "X", Concatenate);
                if(ugrid->GetFieldData()->GetArray(Concatenate))        return 0;
                //
                Concatenate[256];
                this->ConcatenateStrings("Step", stepnum, nodesetname, "Force", "Y", Concatenate);
                if(ugrid->GetFieldData()->GetArray(Concatenate))        return 0;
                //
                Concatenate[256];
                this->ConcatenateStrings("Step", stepnum, nodesetname, "Force", "Z", Concatenate);
                if(ugrid->GetFieldData()->GetArray(Concatenate))        return 0;
                //
                Concatenate[256];
                this->ConcatenateStrings("Step", stepnum, nodesetname, "Displacement", "X", Concatenate);
                if(ugrid->GetFieldData()->GetArray(Concatenate))        return 0;
                //
                Concatenate[256];
                this->ConcatenateStrings("Step", stepnum, nodesetname, "Displacement", "Y", Concatenate);
                if(ugrid->GetFieldData()->GetArray(Concatenate))        return 0;
                //
                Concatenate[256];
                this->ConcatenateStrings("Step", stepnum, nodesetname, "Displacement", "Z", Concatenate);
                if(ugrid->GetFieldData()->GetArray(Concatenate))        return 0;
                //
                Concatenate[256];
                this->ConcatenateStrings("Step", stepnum, nodesetname, "Rotation", "X", Concatenate);
                if(ugrid->GetFieldData()->GetArray(Concatenate))        return 0;
                //
                Concatenate[256];
                this->ConcatenateStrings("Step", stepnum, nodesetname, "Rotation", "Y", Concatenate);
                if(ugrid->GetFieldData()->GetArray(Concatenate))        return 0;
                //
                Concatenate[256];
                this->ConcatenateStrings("Step", stepnum, nodesetname, "Rotation", "Z", Concatenate);
                if(ugrid->GetFieldData()->GetArray(Concatenate))        return 0;
                //
                Concatenate[256];
                this->ConcatenateStrings("Step", stepnum, nodesetname, "Moment", "X", Concatenate);
                if(ugrid->GetFieldData()->GetArray(Concatenate))        return 0;
                //
                Concatenate[256];
                this->ConcatenateStrings("Step", stepnum, nodesetname, "Moment", "Y", Concatenate);
                if(ugrid->GetFieldData()->GetArray(Concatenate))        return 0;
                //
                Concatenate[256];
                this->ConcatenateStrings("Step", stepnum, nodesetname, "Moment", "Z", Concatenate);
                if(ugrid->GetFieldData()->GetArray(Concatenate))        return 0;
        }
        return 1;
}
//-----------------------------------------------------------------------------------------
void vtkKWMimxAssignBoundaryConditionsGroup::ViewBoundaryConditionsCallback(int Mode)
{
        if(!Mode)
        {
                if(this->GlyphActor)
                        this->GetMimxMainWindow()->GetRenderWidget()->RemoveViewProp(this->GlyphActor);
                this->GetMimxMainWindow()->GetRenderWidget()->Render();
                return;
        }
        const char *meshName = this->ObjectListComboBox->GetWidget()->GetValue();

        vtkMimxErrorCallback *callback = this->GetMimxMainWindow()->GetErrorCallback();
        // check if the given step number has any boundary conditions
        if(!strcmp(meshName,""))
        {
                callback->ErrorMessage("FE Mesh selection required");
                return;
        }
        const char *nodeSetName = this->NodeSetComboBox->GetWidget()->GetValue();
        const char *boundCondType = this->BoundaryConditionTypeComboBox->GetWidget()->GetValue();
        const char *stepNum = this->StepNumberComboBox->GetWidget()->GetValue();
        const char *Direction = this->ViewDirectionComboBox->GetWidget()->GetValue();

        char boundname[256];

        this->ConcatenateStrings("Step", stepNum, nodeSetName, boundCondType, Direction, boundname);

        vtkMimxMeshActor *meshActor = vtkMimxMeshActor::SafeDownCast(this->FEMeshList->GetItem(
                this->ObjectListComboBox->GetWidget()->GetValueIndex(meshName)));

        vtkUnstructuredGrid *ugrid = meshActor->GetDataSet();

        if(!ugrid->GetFieldData()->GetArray(boundname))
        {
                callback->ErrorMessage("Boundary condition data not found, check for all the selections made");
                return;
        }
        if(!meshActor->GetIsAverageEdgeLengthCalculated())      meshActor->CalculateAverageEdgeLength();

        vtkPointSet *pointSet = meshActor->GetPointSetOfNodeSet(nodeSetName);

        if(!pointSet)   return;

        double glyphSize = meshActor->GetAverageEdgeLength();

        vtkSphereSource *sphereSource = vtkSphereSource::New();
        sphereSource->SetRadius(glyphSize/8.0);

        if(!this->GlyphActor)   this->GlyphActor = vtkActor::New();

        vtkGlyph3D *Glyph = vtkGlyph3D::New();
        Glyph->SetInput(pointSet);
        Glyph->SetSource(sphereSource->GetOutput());
        Glyph->Update();
        
        vtkPolyDataMapper *glyphMapper = vtkPolyDataMapper::New();
        glyphMapper->SetInput(Glyph->GetOutput());

        this->GlyphActor->SetMapper(glyphMapper);
        this->GlyphActor->GetProperty()->SetColor(1.0, 0.0, 0.0);
        this->GetMimxMainWindow()->GetRenderWidget()->AddViewProp(this->GlyphActor);
        this->GetMimxMainWindow()->GetRenderWidget()->Render();
}
