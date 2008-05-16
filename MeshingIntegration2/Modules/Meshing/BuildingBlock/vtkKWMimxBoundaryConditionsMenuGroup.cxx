/*=========================================================================

Program:   MIMX Meshing Toolkit
Module:    $RCSfile: vtkKWMimxBoundaryConditionsMenuGroup.cxx,v $
Language:  C++
Date:      $Date: 2008/04/22 22:55:57 $
Version:   $Revision: 1.4 $

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

#include "vtkKWMimxBoundaryConditionsMenuGroup.h"
#include "vtkKWMimxMainNotebook.h"

#include "vtkMimxErrorCallback.h"
#include "vtkKWMimxConstMatPropElSetGroup.h"

#include "vtkMimxActorBase.h"
#include "vtkKWMimxDeleteObjectGroup.h"
#include "vtkMimxImageActor.h"

#include "vtkActor.h"
#include "vtkMath.h"
#include "vtkObjectFactory.h"
#include "vtkPolyDataMapper.h"
#include "vtkPolyDataReader.h"
#include "vtkSTLReader.h"
#include "vtkExecutive.h"
#include "vtkCommand.h"
#include "vtkCallbackCommand.h"
#include "vtkKWMessageDialog.h"
#include "vtkMimxErrorCallback.h"

#include "vtkKWApplication.h"
#include "vtkKWCheckButton.h"
#include "vtkKWFileBrowserDialog.h"
#include "vtkKWEvent.h"
#include "vtkKWFrame.h"
#include "vtkKWFrameWithLabel.h"
#include "vtkKWFrameWithScrollbar.h"
#include "vtkKWIcon.h"
#include "vtkKWInternationalization.h"
#include "vtkKWLabel.h"
#include "vtkKWMenu.h"
#include "vtkKWMenuButton.h"
#include "vtkKWMenuButtonWithLabel.h"
#include "vtkKWNotebook.h"
#include "vtkKWOptions.h"
#include "vtkKWRenderWidget.h"
#include "vtkKWTkUtilities.h"
#include "vtkKWMimxEvaluateMeshQualityGroup.h"
#include "vtkKWMimxMainUserInterfacePanel.h"
#include "vtkKWMimxAssignBoundaryConditionsGroup.h"
#include "vtkKWMimxDeleteBoundaryConditionsGroup.h"

#include <vtksys/stl/list>
#include <vtksys/stl/algorithm>

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkKWMimxBoundaryConditionsMenuGroup);
vtkCxxRevisionMacro(vtkKWMimxBoundaryConditionsMenuGroup, "$Revision: 1.4 $");

//----------------------------------------------------------------------------
vtkKWMimxBoundaryConditionsMenuGroup::vtkKWMimxBoundaryConditionsMenuGroup()
{
        this->ObjectMenuButton = vtkKWMenuButtonWithLabel::New();
        this->OperationMenuButton = NULL;
        this->FEMeshList = NULL;
        this->TypeMenuButton = NULL;
        this->AssignBoundaryConditionsGroup = NULL;
        this->DeleteBoundaryConditionsGroup = NULL;
}
//----------------------------------------------------------------------------
vtkKWMimxBoundaryConditionsMenuGroup::~vtkKWMimxBoundaryConditionsMenuGroup()
{
          this->ObjectMenuButton->Delete();
          if(this->OperationMenuButton)
                this->OperationMenuButton->Delete();
          if(this->TypeMenuButton)
                this->TypeMenuButton->Delete();
          if(this->AssignBoundaryConditionsGroup)
                  this->AssignBoundaryConditionsGroup->Delete();
          if(this->DeleteBoundaryConditionsGroup)
                  this->DeleteBoundaryConditionsGroup->Delete();
}
//----------------------------------------------------------------------------
void vtkKWMimxBoundaryConditionsMenuGroup::CreateWidget()
{
        if(this->IsCreated())
        {
                vtkErrorMacro("class already created");
                return;
        }
        this->Superclass::CreateWidget();
        // add menu button with options for various Object
        // for surface
  if(!this->MainFrame)
    this->MainFrame = vtkKWFrameWithLabel::New();

  this->MainFrame->SetParent(this->GetParent());
  this->MainFrame->Create();
  this->MainFrame->SetLabelText("Boundary Condition Operations");

  this->GetApplication()->Script(
    "pack %s -side top -anchor nw -expand n -padx 2 -pady 0 -fill x", 
    this->MainFrame->GetWidgetName());

        // declare operations menu
        if(!this->OperationMenuButton)  
                this->OperationMenuButton = vtkKWMenuButtonWithLabel::New();
        this->OperationMenuButton->SetParent(this->MainFrame->GetFrame());
        this->OperationMenuButton->Create();
        this->OperationMenuButton->SetBorderWidth(0);
        this->OperationMenuButton->SetReliefToGroove();
        this->OperationMenuButton->SetLabelText("Operation :");
        this->OperationMenuButton->SetPadX(2);
        this->OperationMenuButton->SetPadY(2);
        this->OperationMenuButton->SetWidth(30);
        this->OperationMenuButton->SetLabelWidth(10);
        this->GetApplication()->Script("pack %s -side top -anchor nw -expand y -padx 2 -pady 5 -fill x", 
                this->OperationMenuButton->GetWidgetName());
        this->OperationMenuButton->SetEnabled(1);
        this->OperationMenuButton->GetWidget()->GetMenu()->AddRadioButton(
                "Assign",this, "AssignBoundaryConditionsCallback");
        this->OperationMenuButton->GetWidget()->GetMenu()->AddRadioButton(
                "Delete",this, "DeleteBoundaryConditionsCallback");

        //this->OperationMenuButton->GetWidget()->GetMenu()->AddRadioButton(
        //      "Mesh Improvement",this, "MeshImprovementCallback");
  //this->OperationMenuButton->GetWidget()->GetMenu()->AddRadioButton(
        //      "Improve",this, "ImproveMeshCallback");
                
        // declare type menu
        if(!this->TypeMenuButton)       
                this->TypeMenuButton = vtkKWMenuButtonWithLabel::New();
        this->TypeMenuButton->SetParent(this->MainFrame->GetFrame());
        this->TypeMenuButton->Create();
        this->TypeMenuButton->SetBorderWidth(0);
        this->TypeMenuButton->SetReliefToGroove();
        this->TypeMenuButton->SetLabelText("Type :");
        this->TypeMenuButton->SetPadX(2);
        this->TypeMenuButton->SetPadY(2);
        this->TypeMenuButton->SetWidth(30);
        this->TypeMenuButton->SetLabelWidth(10);
        this->GetApplication()->Script("pack %s -side top -anchor nw -expand y -padx 2 -pady 5 -fill x", 
                this->TypeMenuButton->GetWidgetName());
        this->TypeMenuButton->SetEnabled(0);
        //this->MessageDialog->SetApplication(this->GetApplication());
}
//----------------------------------------------------------------------------
void vtkKWMimxBoundaryConditionsMenuGroup::Update()
{
        this->UpdateEnableState();
}
//----------------------------------------------------------------------------
void vtkKWMimxBoundaryConditionsMenuGroup::UpdateEnableState()
{
        this->Superclass::UpdateEnableState();
}

//-----------------------------------------------------------------------------
void vtkKWMimxBoundaryConditionsMenuGroup::AssignBoundaryConditionsCallback()
{
//  this->HideAllDialogBoxes();
  /*this->TypeMenuButton->SetLabelText("Type :");
  this->TypeMenuButton->GetWidget()->GetMenu()->DeleteAllItems();
  this->TypeMenuButton->GetWidget()->SetValue("");
  this->TypeMenuButton->GetWidget()->GetMenu()->AddRadioButton(
    "Assign",this, "AssignCallback");*/
        if (!this->AssignBoundaryConditionsGroup)
        {
                this->AssignBoundaryConditionsGroup = vtkKWMimxAssignBoundaryConditionsGroup::New();
                this->AssignBoundaryConditionsGroup->SetApplication(this->GetApplication());
                this->AssignBoundaryConditionsGroup->SetParent(this->GetParent() /*this->MainFrame->GetFrame()*/);
                this->AssignBoundaryConditionsGroup->SetFEMeshList(this->FEMeshList);
                this->AssignBoundaryConditionsGroup->SetMimxMainWindow(this->GetMimxMainWindow());
                this->AssignBoundaryConditionsGroup->SetViewProperties(
                        this->GetMimxMainWindow()->GetViewProperties());
                this->AssignBoundaryConditionsGroup->SetMenuGroup(this);
                //      this->SetMenuButtonsEnabled(0);
                this->AssignBoundaryConditionsGroup->Create();
        }
        else
        {
                this->AssignBoundaryConditionsGroup->UpdateObjectLists();
        }
        this->SetMenuButtonsEnabled(0);
        this->GetMimxMainWindow()->GetMainUserInterfacePanel()->GetMimxMainNotebook()->SetEnabled(0);

        this->GetApplication()->Script(
                "pack %s -side top -anchor nw -expand n -padx 0 -pady 2 -fill x", 
                this->AssignBoundaryConditionsGroup->GetMainFrame()->GetWidgetName());


}
//-----------------------------------------------------------------------------
void vtkKWMimxBoundaryConditionsMenuGroup::UpdateObjectLists()
{
        //if(this->ConstMatPropElSetGroup)
        //      this->ConstMatPropElSetGroup->UpdateObjectLists();
}
//------------------------------------------------------------------------------
void vtkKWMimxBoundaryConditionsMenuGroup::DeleteBoundaryConditionsCallback()
{
        if (!this->DeleteBoundaryConditionsGroup)
        {
                this->DeleteBoundaryConditionsGroup = vtkKWMimxDeleteBoundaryConditionsGroup::New();
                this->DeleteBoundaryConditionsGroup->SetApplication(this->GetApplication());
                this->DeleteBoundaryConditionsGroup->SetParent(this->GetParent() /*this->MainFrame->GetFrame()*/);
                this->DeleteBoundaryConditionsGroup->SetFEMeshList(this->FEMeshList);
                this->DeleteBoundaryConditionsGroup->SetMimxMainWindow(this->GetMimxMainWindow());
                this->DeleteBoundaryConditionsGroup->SetViewProperties(
                        this->GetMimxMainWindow()->GetViewProperties());
                this->DeleteBoundaryConditionsGroup->SetMenuGroup(this);
                //      this->SetMenuButtonsEnabled(0);
                this->DeleteBoundaryConditionsGroup->Create();
        }
        else
        {
                this->DeleteBoundaryConditionsGroup->UpdateObjectLists();
        }
        this->SetMenuButtonsEnabled(0);
        this->GetMimxMainWindow()->GetMainUserInterfacePanel()->GetMimxMainNotebook()->SetEnabled(0);

        this->GetApplication()->Script(
                "pack %s -side top -anchor nw -expand n -padx 0 -pady 2 -fill x", 
                this->DeleteBoundaryConditionsGroup->GetMainFrame()->GetWidgetName());
}
//----------------------------------------------------------------------------
void vtkKWMimxBoundaryConditionsMenuGroup::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
//---------------------------------------------------------------------------
