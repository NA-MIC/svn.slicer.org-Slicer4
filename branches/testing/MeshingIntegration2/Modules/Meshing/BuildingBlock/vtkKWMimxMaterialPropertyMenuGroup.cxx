/*=========================================================================

Program:   MIMX Meshing Toolkit
Module:    $RCSfile: vtkKWMimxMaterialPropertyMenuGroup.cxx,v $
Language:  C++
Date:      $Date: 2008/04/20 14:39:04 $
Version:   $Revision: 1.2 $

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

#include "vtkKWMimxMaterialPropertyMenuGroup.h"
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
#include "vtkKWMimxApplyFEMeshMaterialPropertiesFromImageGroup.h"
#include "vtkKWMimxReBinFEMeshMaterialPropertiesFromImageGroup.h"

#include <vtksys/stl/list>
#include <vtksys/stl/algorithm>

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkKWMimxMaterialPropertyMenuGroup);
vtkCxxRevisionMacro(vtkKWMimxMaterialPropertyMenuGroup, "$Revision: 1.2 $");

//----------------------------------------------------------------------------
vtkKWMimxMaterialPropertyMenuGroup::vtkKWMimxMaterialPropertyMenuGroup()
{
        this->ObjectMenuButton = vtkKWMenuButtonWithLabel::New();
        this->OperationMenuButton = NULL;
        this->TypeMenuButton = NULL;
        this->ConstMatPropElSetGroup = NULL;
        this->ImageMatPropGroup = NULL;
        this->ReBinImageMatPropGroup = NULL;
        this->ImageList = NULL;
}
//----------------------------------------------------------------------------
vtkKWMimxMaterialPropertyMenuGroup::~vtkKWMimxMaterialPropertyMenuGroup()
{
          this->ObjectMenuButton->Delete();
          if(this->OperationMenuButton)
                this->OperationMenuButton->Delete();
          if(this->TypeMenuButton)
                this->TypeMenuButton->Delete();
          if(this->ConstMatPropElSetGroup)
                  this->ConstMatPropElSetGroup->Delete();
          if(this->ImageMatPropGroup)
                  this->ImageMatPropGroup->Delete();
          if(this->ReBinImageMatPropGroup)
                  this->ReBinImageMatPropGroup->Delete();
 }
//----------------------------------------------------------------------------
void vtkKWMimxMaterialPropertyMenuGroup::CreateWidget()
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
  this->MainFrame->SetLabelText("Material Property Operations");

  this->GetApplication()->Script(
    "pack %s -side top -anchor nw -expand n -padx 2 -pady 6 -fill x", 
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
                "Assign",this, "AssignMaterialPropertiesCallback");
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
void vtkKWMimxMaterialPropertyMenuGroup::Update()
{
        this->UpdateEnableState();
}
//----------------------------------------------------------------------------
void vtkKWMimxMaterialPropertyMenuGroup::UpdateEnableState()
{
        this->Superclass::UpdateEnableState();
}

//-----------------------------------------------------------------------------
void vtkKWMimxMaterialPropertyMenuGroup::AssignMaterialPropertiesCallback()
{
//  this->HideAllDialogBoxes();
  this->TypeMenuButton->SetLabelText("Type :");
  this->TypeMenuButton->GetWidget()->GetMenu()->DeleteAllItems();
  this->TypeMenuButton->GetWidget()->SetValue("");
  this->TypeMenuButton->GetWidget()->GetMenu()->AddRadioButton(
    "Constant Properties",this, "AssignMaterialPropertiesElementSetCallback");
  this->TypeMenuButton->GetWidget()->GetMenu()->AddRadioButton(
          "Image Based",this, "AssignImageBasedMaterialPropertiesCallback");
  this->TypeMenuButton->GetWidget()->GetMenu()->AddRadioButton(
          "Re-Bin",this, "ReBinImageBasedMaterialPropertiesCallback");
}
//-----------------------------------------------------------------------------
void vtkKWMimxMaterialPropertyMenuGroup::AssignMaterialPropertiesElementSetCallback()
{
        if (!this->ConstMatPropElSetGroup)
        {
                this->ConstMatPropElSetGroup = vtkKWMimxConstMatPropElSetGroup::New();
                this->ConstMatPropElSetGroup->SetApplication(this->GetApplication());
                this->ConstMatPropElSetGroup->SetParent(this->GetParent() /*this->MainFrame->GetFrame()*/);
                this->ConstMatPropElSetGroup->SetSurfaceList(this->SurfaceList);
                this->ConstMatPropElSetGroup->SetBBoxList(this->BBoxList);
                this->ConstMatPropElSetGroup->SetFEMeshList(this->FEMeshList);
                this->ConstMatPropElSetGroup->SetMimxMainWindow(this->GetMimxMainWindow());
                this->ConstMatPropElSetGroup->SetViewProperties(
                        this->GetMimxMainWindow()->GetViewProperties());
                this->ConstMatPropElSetGroup->SetMenuGroup(this);
                //      this->SetMenuButtonsEnabled(0);
                this->ConstMatPropElSetGroup->Create();
        }
        else
        {
                this->ConstMatPropElSetGroup->UpdateObjectLists();
        }
        this->SetMenuButtonsEnabled(0);
        this->GetMimxMainWindow()->GetMainUserInterfacePanel()->GetMimxMainNotebook()->SetEnabled(0);

        this->GetApplication()->Script(
                "pack %s -side top -anchor nw -expand n -padx 0 -pady 2 -fill x", 
                this->ConstMatPropElSetGroup->GetMainFrame()->GetWidgetName());
}
//-----------------------------------------------------------------------------
void vtkKWMimxMaterialPropertyMenuGroup::UpdateObjectLists()
{
        if(this->ConstMatPropElSetGroup)
                this->ConstMatPropElSetGroup->UpdateObjectLists();
}
//----------------------------------------------------------------------------
void vtkKWMimxMaterialPropertyMenuGroup::AssignImageBasedMaterialPropertiesCallback()
{
                if(!this->ImageMatPropGroup)
                {
                        this->ImageMatPropGroup = vtkKWMimxApplyFEMeshMaterialPropertiesFromImageGroup::New();
                        this->ImageMatPropGroup->SetParent(this->GetParent() /*this->MainFrame->GetFrame()*/);
                        //              this->EditNodeElementNumbersGroup->SetSurfaceList(this->SurfaceList);
                        this->ImageMatPropGroup->SetFEMeshList(this->FEMeshList);
                        this->ImageMatPropGroup->SetImageList(this->ImageList);
                        this->ImageMatPropGroup->SetMenuGroup(this);
                        this->ImageMatPropGroup->SetMimxMainWindow(this->GetMimxMainWindow());
                        this->ImageMatPropGroup->SetViewProperties(
                                this->GetMimxMainWindow()->GetViewProperties());
        //              this->ImageMatPropGroup->SetViewProperties(this->MimxViewProperties);
                        this->ImageMatPropGroup->Create();
                }
                else
                {
                        this->ImageMatPropGroup->UpdateObjectLists();
                }
                this->SetMenuButtonsEnabled(0);
                  this->GetMimxMainWindow()->GetMainUserInterfacePanel()->GetMimxMainNotebook()->SetEnabled(0);
        
                this->GetApplication()->Script("pack %s -side top -anchor nw -expand n -padx 2 -pady 5 -fill x", 
                        this->ImageMatPropGroup->GetMainFrame()->GetWidgetName());
}
//------------------------------------------------------------------------------------------------------
void vtkKWMimxMaterialPropertyMenuGroup::ReBinImageBasedMaterialPropertiesCallback()
{
        if(!this->ReBinImageMatPropGroup)
        {
                this->ReBinImageMatPropGroup = vtkKWMimxReBinFEMeshMaterialPropertiesFromImageGroup::New();
                this->ReBinImageMatPropGroup->SetParent(this->GetParent() /*this->MainFrame->GetFrame()*/);
                //              this->EditNodeElementNumbersGroup->SetSurfaceList(this->SurfaceList);
                this->ReBinImageMatPropGroup->SetFEMeshList(this->FEMeshList);
                this->ReBinImageMatPropGroup->SetMenuGroup(this);
                this->ReBinImageMatPropGroup->SetMimxMainWindow(this->GetMimxMainWindow());
                this->ReBinImageMatPropGroup->SetViewProperties(
                        this->GetMimxMainWindow()->GetViewProperties());
                //              this->ImageMatPropGroup->SetViewProperties(this->MimxViewProperties);
                this->ReBinImageMatPropGroup->Create();
        }
        else
        {
                this->ReBinImageMatPropGroup->UpdateObjectLists();
        }
        this->SetMenuButtonsEnabled(0);
        this->GetMimxMainWindow()->GetMainUserInterfacePanel()->GetMimxMainNotebook()->SetEnabled(0);

        this->GetApplication()->Script("pack %s -side top -anchor nw -expand n -padx 2 -pady 5 -fill x", 
                this->ReBinImageMatPropGroup->GetMainFrame()->GetWidgetName());
}
//----------------------------------------------------------------------------
void vtkKWMimxMaterialPropertyMenuGroup::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
//---------------------------------------------------------------------------
