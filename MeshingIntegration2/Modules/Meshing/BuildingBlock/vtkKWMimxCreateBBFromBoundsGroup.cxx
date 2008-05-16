/*=========================================================================

Program:   MIMX Meshing Toolkit
Module:    $RCSfile: vtkKWMimxCreateBBFromBoundsGroup.cxx,v $
Language:  C++
Date:      $Date: 2008/04/20 14:39:04 $
Version:   $Revision: 1.26 $

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

#include "vtkKWMimxCreateBBFromBoundsGroup.h"
#include "vtkKWMimxMainWindow.h"
#include "vtkKWMimxMainMenuGroup.h"
#include "vtkMimxErrorCallback.h"

#include "vtkLinkedListWrapper.h"

#include "vtkActor.h"
#include "vtkMimxBoundingBoxSource.h"
#include "vtkMimxSurfacePolyDataActor.h"
#include "vtkPolyData.h"
#include "vtkProperty.h"
#include "vtkMimxUnstructuredGridActor.h"
#include "vtkUnstructuredGrid.h"
#include "vtkRenderer.h"


#include "vtkKWApplication.h"
#include "vtkKWFileBrowserDialog.h"
#include "vtkKWEvent.h"
#include "vtkKWFrame.h"
#include "vtkKWFrameWithLabel.h"
#include "vtkKWLabel.h"
#include "vtkKWMenu.h"
#include "vtkKWMenuButton.h"
#include "vtkKWMenuButtonWithLabel.h"
#include "vtkKWNotebook.h"
#include "vtkKWOptions.h"
#include "vtkKWRenderWidget.h"
#include "vtkKWTkUtilities.h"
#include "vtkObjectFactory.h"
#include "vtkKWComboBoxWithLabel.h"
#include "vtkKWComboBox.h"
#include "vtkKWPushButton.h"

#include "vtkKWMimxBBMenuGroup.h"
#include "vtkKWMimxSurfaceMenuGroup.h"
#include "vtkKWMimxMainNotebook.h"
#include "vtkKWMimxMainUserInterfacePanel.h"

#include <vtksys/stl/list>
#include <vtksys/stl/algorithm>

// define the option types
#define VTK_KW_OPTION_NONE         0
#define VTK_KW_OPTION_LOAD                 1

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkKWMimxCreateBBFromBoundsGroup);
vtkCxxRevisionMacro(vtkKWMimxCreateBBFromBoundsGroup, "$Revision: 1.26 $");

//----------------------------------------------------------------------------
vtkKWMimxCreateBBFromBoundsGroup::vtkKWMimxCreateBBFromBoundsGroup()
{
  this->ObjectListComboBox = NULL;
  this->SurfaceMenuGroup = NULL;
}

//----------------------------------------------------------------------------
vtkKWMimxCreateBBFromBoundsGroup::~vtkKWMimxCreateBBFromBoundsGroup()
{
  if(this->ObjectListComboBox)
     this->ObjectListComboBox->Delete();
 }
//----------------------------------------------------------------------------
void vtkKWMimxCreateBBFromBoundsGroup::CreateWidget()
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
  this->MainFrame->SetLabelText("Create Building Block From Bounds");

  this->GetApplication()->Script(
    "pack %s -side top -anchor nw -expand n -padx 2 -pady 0 -fill x", 
    this->MainFrame->GetWidgetName());

  ObjectListComboBox->SetParent(this->MainFrame->GetFrame());
  ObjectListComboBox->Create();
  ObjectListComboBox->SetLabelText("Surface : ");
  ObjectListComboBox->SetLabelWidth( 15 );
  ObjectListComboBox->GetWidget()->ReadOnlyOn();
  ObjectListComboBox->GetWidget()->SetCommand(this, "SelectionChangedCallback");
  
  this->GetApplication()->Script(
    "pack %s -side top -anchor nw -expand y -padx 2 -pady 6 -fill x", 
    ObjectListComboBox->GetWidgetName());

  this->ApplyButton->SetParent(this->MainFrame->GetFrame());
  this->ApplyButton->Create();
  this->ApplyButton->SetText("Apply");
  this->ApplyButton->SetCommand(this, "CreateBBFromBoundsApplyCallback");
  this->GetApplication()->Script(
          "pack %s -side left -anchor nw -expand y -padx 20 -pady 6", 
          this->ApplyButton->GetWidgetName());

/*  this->DoneButton->SetParent(this->MainFrame->GetFrame());
  this->DoneButton->Create();
  this->DoneButton->SetText("Done");
  this->DoneButton->SetCommand(this, "CreateBBFromBoundsDoneCallback");
  this->GetApplication()->Script(
    "pack %s -side left -anchor nw -expand y -padx 20 -pady 6", 
    this->DoneButton->GetWidgetName());
*/
  this->CancelButton->SetParent(this->MainFrame->GetFrame());
  this->CancelButton->Create();
  this->CancelButton->SetText("Cancel");
  this->CancelButton->SetCommand(this, "CreateBBFromBoundsCancelCallback");
  this->GetApplication()->Script(
    "pack %s -side right -anchor ne -expand y -padx 20 -pady 6", 
    this->CancelButton->GetWidgetName());

}
//----------------------------------------------------------------------------
void vtkKWMimxCreateBBFromBoundsGroup::Update()
{
        this->UpdateEnableState();
}
//---------------------------------------------------------------------------
void vtkKWMimxCreateBBFromBoundsGroup::UpdateEnableState()
{
        this->UpdateObjectLists();
        this->Superclass::UpdateEnableState();
}
//----------------------------------------------------------------------------
int vtkKWMimxCreateBBFromBoundsGroup::CreateBBFromBoundsApplyCallback()
{
        vtkMimxErrorCallback *callback = this->GetMimxMainWindow()->GetErrorCallback();
//      callback->SetState(0);
  if(!strcmp(this->ObjectListComboBox->GetWidget()->GetValue(),""))
  {
                callback->ErrorMessage("Object not chosen");
                return 0;
  }
    vtkKWComboBox *combobox = this->ObjectListComboBox->GetWidget();
    const char *name = combobox->GetValue();
        int num = combobox->GetValueIndex(name);
        if(num < 0 || num > combobox->GetNumberOfValues()-1)
        {
                callback->ErrorMessage("Choose valid surface");
                combobox->SetValue("");
                return 0;
        }
    vtkPolyData *polydata = vtkMimxSurfacePolyDataActor::SafeDownCast(
                this->SurfaceList->GetItem(combobox->GetValueIndex(name)))->GetDataSet();
        callback->SetState(0);
    vtkMimxBoundingBoxSource *bbox = vtkMimxBoundingBoxSource::New();
    bbox->SetSource(polydata);
        bbox->AddObserver(vtkCommand::ErrorEvent, callback, 1.0);
    bbox->Update();
    if (!callback->GetState())
    {
      this->BBoxList->AppendItem(vtkMimxUnstructuredGridActor::New());
          // for do and undo tree
          this->DoUndoTree->AppendItem(new Node);
          int currentitem = this->BBoxList->GetNumberOfItems()-1;
          this->DoUndoTree->GetItem(currentitem)->Parent = NULL;
          this->DoUndoTree->GetItem(currentitem)->Child = NULL;
          this->DoUndoTree->GetItem(currentitem)->Data = 
                  vtkMimxUnstructuredGridActor::SafeDownCast(
                  this->BBoxList->GetItem(this->BBoxList->GetNumberOfItems()-1));
          ////    
          this->BBoxList->GetItem(this->BBoxList->GetNumberOfItems()-1)->
                   SetDataType(ACTOR_BUILDING_BLOCK);
      vtkMimxUnstructuredGridActor::SafeDownCast(this->BBoxList->GetItem(
        this->BBoxList->GetNumberOfItems()-1))->GetDataSet()->DeepCopy(bbox->GetOutput());
      this->Count++;
    /* Assign Mesh Seeds - Should be an Application Configured Parameter*/
    vtkMimxUnstructuredGridActor *ugridactor = 
        vtkMimxUnstructuredGridActor::SafeDownCast(this->BBoxList->GetItem(
        this->BBoxList->GetNumberOfItems()-1));
    ugridactor->MeshSeedFromAverageElementLength( 1.0, 1.0, 1.0 );
   
    
        
      vtkMimxUnstructuredGridActor::SafeDownCast(this->BBoxList->GetItem(
        this->BBoxList->GetNumberOfItems()-1))->SetObjectName("BBFromBounds_",Count);
      vtkMimxUnstructuredGridActor::SafeDownCast(this->BBoxList->GetItem(
        this->BBoxList->GetNumberOfItems()-1))->GetDataSet()->Modified();
      this->GetMimxMainWindow()->GetRenderWidget()->AddViewProp(
        this->BBoxList->GetItem(this->BBoxList->GetNumberOfItems()-1)->GetActor());
      this->GetMimxMainWindow()->GetRenderWidget()->Render();
      this->GetMimxMainWindow()->GetRenderWidget()->ResetCamera();
          this->GetMimxMainWindow()->GetViewProperties()->AddObjectList(
                  this->BBoxList->GetItem(this->BBoxList->GetNumberOfItems()-1));
          this->CreateBBFromBoundsCancelCallback();
          bbox->Delete();
          
          this->GetMimxMainWindow()->SetStatusText("Created Building Block From Bounds");
          
          return 1;
    }
        bbox->Delete();
        return 0;
}
//----------------------------------------------------------------------------
void vtkKWMimxCreateBBFromBoundsGroup::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
//----------------------------------------------------------------------------
void vtkKWMimxCreateBBFromBoundsGroup::CreateBBFromBoundsCancelCallback()
{
//  this->MainFrame->UnpackChildren();
  this->GetApplication()->Script("pack forget %s", this->MainFrame->GetWidgetName());
  this->MenuGroup->SetMenuButtonsEnabled(1);
//  this->SurfaceMenuGroup->SetEnabled(1);
    this->GetMimxMainWindow()->GetMainUserInterfacePanel()->GetMimxMainNotebook()->SetEnabled(1);
}
//------------------------------------------------------------------------------
void vtkKWMimxCreateBBFromBoundsGroup::UpdateObjectLists()
{
  this->ObjectListComboBox->GetWidget()->DeleteAllValues();
  
  int defaultItem = -1;
  for (int i = 0; i < this->SurfaceList->GetNumberOfItems(); i++)
  {
    ObjectListComboBox->GetWidget()->AddValue(
      this->SurfaceList->GetItem(i)->GetFileName());
      
    int viewedItem = this->GetMimxMainWindow()->GetRenderWidget()->GetRenderer()->HasViewProp(
                        this->SurfaceList->GetItem(i)->GetActor());
    if ( (defaultItem == -1) && ( viewedItem ) )
                {
                  defaultItem = i;
                }
  }
  
  if (defaultItem != -1)
  {
    ObjectListComboBox->GetWidget()->SetValue(
          this->SurfaceList->GetItem(defaultItem)->GetFileName());
  }
}
//------------------------------------------------------------------------------
void vtkKWMimxCreateBBFromBoundsGroup::SelectionChangedCallback(const char *dummy)
{
        //if(strcmp(this->ObjectListComboBox->GetWidget()->GetValue(),""))
        //{
        //      vtkKWComboBox *combobox = this->ObjectListComboBox->GetWidget();
        //      const char *name = combobox->GetValue();
        //      vtkMimxSurfacePolyDataActor *actor = vtkMimxSurfacePolyDataActor::SafeDownCast(
        //              this->SurfaceList->GetItem(combobox->GetValueIndex(name)));
        //      for (int i=0; i<3; i++)
        //      {
        //              double prevcolor[3], currcolor[3];
        //              actor->GetActor()->GetProperty()->GetColor(prevcolor);
        //              currcolor[0] = prevcolor[0]/2.0;
        //              currcolor[1] = prevcolor[1]/2.0 + 0.25;
        //              currcolor[2] = prevcolor[2]/2.0 + 0.5;
        //              actor->GetActor()->GetProperty()->SetColor(currcolor);
        //              this->GetMimxMainWindow()->GetRenderWidget()->Render();
        //              Sleep(100);
        //              actor->GetActor()->GetProperty()->SetColor(prevcolor);
        //              this->GetMimxMainWindow()->GetRenderWidget()->Render();
        //      }
        //}
}

void vtkKWMimxCreateBBFromBoundsGroup::CreateBBFromBoundsDoneCallback()
{
        if(this->CreateBBFromBoundsApplyCallback())
                this->CreateBBFromBoundsCancelCallback();
}
