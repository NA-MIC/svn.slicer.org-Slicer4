/*=========================================================================

Program:   MIMX Meshing Toolkit
Module:    $RCSfile: vtkKWMimxCreateFEMeshFromBBGroup.cxx,v $
Language:  C++
Date:      $Date: 2008/04/27 03:34:29 $
Version:   $Revision: 1.34 $

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

#include "vtkKWMimxCreateFEMeshFromBBGroup.h"
#include "vtkKWMimxMainWindow.h"
#include "vtkMimxErrorCallback.h"
#include "vtkKWMimxMainNotebook.h"
#include "vtkMimxApplyNodeElementNumbers.h"

#include "vtkLinkedListWrapper.h"

#include "vtkActor.h"
#include "vtkCellData.h"
#include "vtkIntArray.h"
#include "vtkMimxSurfacePolyDataActor.h"
#include "vtkPolyData.h"
#include "vtkProperty.h"
#include "vtkMimxUnstructuredGridActor.h"
#include "vtkUnstructuredGrid.h"
#include "vtkMimxUnstructuredGridFromBoundingBox.h"
#include "vtkMimxEquivalancePoints.h"
#include "vtkKWMimxNodeElementNumbersGroup.h"
#include "vtkMimxMeshActor.h"

//#include "vtkUnstructuredGridWriter.h"

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
#include "vtkRenderer.h"
#include "vtkKWMimxMainUserInterfacePanel.h"
#include "vtkKWEntryWithLabel.h"
#include "vtkStringArray.h"

#include <vtksys/stl/list>
#include <vtksys/stl/algorithm>

// define the option types
#define VTK_KW_OPTION_NONE         0
#define VTK_KW_OPTION_LOAD                 1

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkKWMimxCreateFEMeshFromBBGroup);
vtkCxxRevisionMacro(vtkKWMimxCreateFEMeshFromBBGroup, "$Revision: 1.34 $");

//----------------------------------------------------------------------------
vtkKWMimxCreateFEMeshFromBBGroup::vtkKWMimxCreateFEMeshFromBBGroup()
{
  this->SurfaceListComboBox = NULL;
  this->BBListComboBox = NULL;
  this->OriginalPosition = NULL;
  this->NodeNumberEntry = NULL;
  this->ElementSetNameEntry = NULL;
  this->ElementNumberEntry = NULL;
  this->NodeElementNumbersGroup = NULL;
}

//----------------------------------------------------------------------------
vtkKWMimxCreateFEMeshFromBBGroup::~vtkKWMimxCreateFEMeshFromBBGroup()
{
  if(this->SurfaceListComboBox)
     this->SurfaceListComboBox->Delete();
  if(this->BBListComboBox)
    this->BBListComboBox->Delete();
  if(this->OriginalPosition)
          this->OriginalPosition->Delete();
        if(this->NodeNumberEntry)
          this->NodeNumberEntry->Delete();
        if(this->ElementSetNameEntry)
          this->ElementSetNameEntry->Delete();
        if(this->ElementNumberEntry)
          this->ElementNumberEntry->Delete();
        if(this->NodeElementNumbersGroup)
                this->NodeElementNumbersGroup->Delete();
 }
//----------------------------------------------------------------------------
void vtkKWMimxCreateFEMeshFromBBGroup::CreateWidget()
{
  if(this->IsCreated())
  {
    vtkErrorMacro("class already created");
    return;
  }

  this->Superclass::CreateWidget();
  if(!this->SurfaceListComboBox)        
  {
     this->SurfaceListComboBox = vtkKWComboBoxWithLabel::New();
  }
  if(!this->BBListComboBox)     
  {
    this->BBListComboBox = vtkKWComboBoxWithLabel::New();
  }
  this->MainFrame->SetParent(this->GetParent());
  this->MainFrame->SetApplication(this->GetApplication());
  this->MainFrame->Create();
  this->MainFrame->SetLabelText("Create Mesh From Building Block");

  this->GetApplication()->Script(
    "pack %s -side top -anchor nw -expand n -padx 2 -pady 0 -fill x", 
    this->MainFrame->GetWidgetName());

  SurfaceListComboBox->SetParent(this->MainFrame->GetFrame());
  SurfaceListComboBox->Create();
  SurfaceListComboBox->SetLabelText("Surface: ");
  SurfaceListComboBox->SetLabelWidth(15);
  SurfaceListComboBox->GetWidget()->ReadOnlyOn();
  SurfaceListComboBox->GetWidget()->SetBalloonHelpString("Surface onto which the resulting mesh is projected");
/*
  int i;
  for (i = 0; i < this->SurfaceList->GetNumberOfItems(); i++)
  {
    SurfaceListComboBox->GetWidget()->AddValue(
      this->SurfaceList->GetItem(i)->GetFileName());
  }
*/
  this->GetApplication()->Script(
    "pack %s -side top -anchor nw -expand y -padx 2 -pady 6 -fill x", 
    SurfaceListComboBox->GetWidgetName());

  BBListComboBox->SetParent(this->MainFrame->GetFrame());
  BBListComboBox->Create();
  BBListComboBox->SetLabelText("Building Block : ");
  BBListComboBox->SetLabelWidth(15);
  BBListComboBox->GetWidget()->ReadOnlyOn();
  BBListComboBox->GetWidget()->SetBalloonHelpString("Building Block for mesh generation");

  this->GetApplication()->Script(
    "pack %s -side top -anchor nw -expand y -padx 2 -pady 6 -fill x", 
    BBListComboBox->GetWidgetName());

  //if(!this->NodeNumberEntry)
         // this->NodeNumberEntry = vtkKWEntryWithLabel::New();

  //this->NodeNumberEntry->SetParent(this->MainFrame->GetFrame());
  //this->NodeNumberEntry->Create();
  //this->NodeNumberEntry->SetWidth(8); 
  //this->NodeNumberEntry->SetLabelText("Node Number : ");
  //this->NodeNumberEntry->SetLabelWidth(15);
  //this->NodeNumberEntry->GetWidget()->SetValueAsInt(1);
  //this->NodeNumberEntry->GetWidget()->SetRestrictValueToInteger();
  //this->GetApplication()->Script(
         // "pack %s -side top -anchor nw -expand n -padx 2 -pady 6 -fill x", 
         // this->NodeNumberEntry->GetWidgetName());


  //if(!this->ElementSetNameEntry)
         // this->ElementSetNameEntry = vtkKWEntryWithLabel::New();

  //this->ElementSetNameEntry->SetParent(this->MainFrame->GetFrame());
  //this->ElementSetNameEntry->Create();
  //this->ElementSetNameEntry->SetWidth(8);
  //this->ElementSetNameEntry->SetLabelText("Element Name : ");
  //this->ElementSetNameEntry->GetWidget()->SetValue("Bone");
  //this->ElementSetNameEntry->SetLabelWidth(15);
  //this->GetApplication()->Script(
         // "pack %s -side top -anchor nw -expand n -padx 2 -pady 6 -fill x", 
         // this->ElementSetNameEntry->GetWidgetName());


  //if(!this->ElementNumberEntry)
         // this->ElementNumberEntry = vtkKWEntryWithLabel::New();

  //this->ElementNumberEntry->SetParent(this->MainFrame->GetFrame());
  //this->ElementNumberEntry->Create();
  //this->ElementNumberEntry->SetWidth(8);
  //this->ElementNumberEntry->SetLabelText("Element Number : ");
  //this->ElementNumberEntry->SetLabelWidth(15);
  //this->ElementNumberEntry->GetWidget()->SetValueAsInt(1);
  //this->ElementNumberEntry->GetWidget()->SetRestrictValueToInteger();
  //this->GetApplication()->Script(
         // "pack %s -side top -anchor nw -expand n -padx 2 -pady 6 -fill x", 
         // this->ElementNumberEntry->GetWidgetName());
         // 

  this->NodeElementNumbersGroup = vtkKWMimxNodeElementNumbersGroup::New();
  this->NodeElementNumbersGroup->SetParent(this->MainFrame->GetFrame());
  this->NodeElementNumbersGroup->Create();
  this->NodeElementNumbersGroup->GetMainFrame()->SetLabelText("Node and Element Numbers");
  this->GetApplication()->Script(
          "pack %s -side top -anchor nw -expand n -padx 2 -pady 6", 
          this->NodeElementNumbersGroup->GetWidgetName());

  this->ApplyButton->SetParent(this->MainFrame->GetFrame());
  this->ApplyButton->Create();
  this->ApplyButton->SetText("Apply");
  this->ApplyButton->SetCommand(this, "CreateFEMeshFromBBApplyCallback");
  this->GetApplication()->Script(
          "pack %s -side left -anchor nw -expand y -padx 5 -pady 6", 
          this->ApplyButton->GetWidgetName());


  this->CancelButton->SetParent(this->MainFrame->GetFrame());
  this->CancelButton->Create();
  this->CancelButton->SetText("Cancel");
  this->CancelButton->SetCommand(this, "CreateFEMeshFromBBCancelCallback");
  this->GetApplication()->Script(
    "pack %s -side right -anchor ne -expand y -padx 5 -pady 6", 
    this->CancelButton->GetWidgetName());

}
//----------------------------------------------------------------------------
void vtkKWMimxCreateFEMeshFromBBGroup::Update()
{
        this->UpdateEnableState();
}
//---------------------------------------------------------------------------
void vtkKWMimxCreateFEMeshFromBBGroup::UpdateEnableState()
{
        this->UpdateObjectLists();
        this->Superclass::UpdateEnableState();
}
//----------------------------------------------------------------------------
int vtkKWMimxCreateFEMeshFromBBGroup::CreateFEMeshFromBBApplyCallback()
{
        vtkMimxErrorCallback *callback = this->GetMimxMainWindow()->GetErrorCallback();

  if(!strcmp(this->SurfaceListComboBox->GetWidget()->GetValue(),""))
  {
        callback->ErrorMessage("Projection surface not chosen");
        return 0;
  }
    if(!strcmp(this->BBListComboBox->GetWidget()->GetValue(),""))
  {
          callback->ErrorMessage("Building Block from which FE mesh to be generated not chosen");
          return 0;
        }
    vtkKWComboBox *combobox = this->SurfaceListComboBox->GetWidget();
    const char *name = combobox->GetValue();

        int num = combobox->GetValueIndex(name);
        if(num < 0 || num > combobox->GetNumberOfValues()-1)
        {
                callback->ErrorMessage("Choose valid Surface");
                combobox->SetValue("");
                return 0;
        }

  vtkPolyData *polydata = vtkMimxSurfacePolyDataActor::SafeDownCast(this->SurfaceList
     ->GetItem(combobox->GetValueIndex(name)))->GetDataSet();

  combobox = this->BBListComboBox->GetWidget();
  name = combobox->GetValue();

        num = combobox->GetValueIndex(name);
        if(num < 0 || num > combobox->GetNumberOfValues()-1)
        {
                callback->ErrorMessage("Choose valid Building-block structure");
                combobox->SetValue("");
                return 0;
        }

  vtkUnstructuredGrid *ugrid = vtkMimxUnstructuredGridActor::SafeDownCast(
      this->BBoxList->GetItem(this->OriginalPosition->GetValue(
          combobox->GetValueIndex(name))))->GetDataSet();
  if(!ugrid->GetCellData()->GetArray("Mesh_Seed"))
        {
                callback->ErrorMessage("Choose building-block structure with mesh seeds assigned");
      return 0;
        }
        
        int nodeNumber = this->NodeElementNumbersGroup->GetNodeNumberEntry()->GetWidget()->GetValueAsInt();
        if (nodeNumber < 1 )
  {
          callback->ErrorMessage("Node numbers must be greater than 0");
          return 0;
  }
  
  int elementNumber = this->NodeElementNumbersGroup->GetElementNumberEntry()->GetWidget()->GetValueAsInt();
  if (elementNumber < 1 )
  {
          callback->ErrorMessage("Element numbers must be greater than 0");
          return 0;
  }
  
  const char *nodesetname = this->NodeElementNumbersGroup->GetNodeSetNameEntry()->
          GetWidget()->GetValue();

  const char *elementsetname = this->NodeElementNumbersGroup->GetElementSetNameEntry()->
          GetWidget()->GetValue();

  if(!strcmp(nodesetname,""))
  {
          callback->ErrorMessage("Node set name cannot be empty");
          return 0;
  }

  if(!strcmp(elementsetname,""))
  {
          callback->ErrorMessage("Element set name cannot be empty");
          return 0;
  }
  
  vtkMimxUnstructuredGridFromBoundingBox *ugridfrombbox = 
      vtkMimxUnstructuredGridFromBoundingBox::New();
  ugridfrombbox->SetBoundingBox(ugrid);
  ugridfrombbox->SetSurface(polydata);
        callback->SetState(0);
        ugridfrombbox->AddObserver(vtkCommand::ErrorEvent, callback, 1.0);
  ugridfrombbox->Update();
  if (!callback->GetState())
  {
                vtkMimxEquivalancePoints *equivalance = vtkMimxEquivalancePoints::New();
                equivalance->SetInput(ugridfrombbox->GetOutput());
                equivalance->Update();
                
                vtkMimxApplyNodeElementNumbers *apply = new vtkMimxApplyNodeElementNumbers;
                apply->SetUnstructuredGrid(equivalance->GetOutput());
                apply->SetNodeSetName(
                        this->NodeElementNumbersGroup->GetNodeSetNameEntry()->GetWidget()->GetValue());
                apply->SetStartingNodeNumber(this->NodeElementNumbersGroup->GetNodeNumberEntry()->
                        GetWidget()->GetValueAsInt());
                apply->ApplyNodeNumbers();

                apply->SetElementSetName(
                        this->NodeElementNumbersGroup->GetElementSetNameEntry()->GetWidget()->GetValue());
                apply->SetStartingElementNumber(this->NodeElementNumbersGroup->
                        GetElementNumberEntry()->GetWidget()->GetValueAsInt());
                apply->ApplyElementNumbers();
                delete apply;
        
    vtkMimxMeshActor *meshActor = vtkMimxMeshActor::New();
    this->FEMeshList->AppendItem( meshActor );
          meshActor->SetDataSet( equivalance->GetOutput() );
                meshActor->SetRenderer( this->GetMimxMainWindow()->GetRenderWidget()->GetRenderer() );
                meshActor->SetInteractor( this->GetMimxMainWindow()->GetRenderWidget()->GetRenderWindowInteractor() );
  
        vtkIntArray *BoundCond = vtkIntArray::New();
        BoundCond->SetNumberOfValues(1);
        BoundCond->SetValue(0,1);
        BoundCond->SetName("Boundary_Condition_Number_Of_Steps");
        equivalance->GetOutput()->GetFieldData()->AddArray(BoundCond);
        BoundCond->Delete();
    this->Count++;
    vtkMimxMeshActor::SafeDownCast(this->FEMeshList->GetItem(
        this->FEMeshList->GetNumberOfItems()-1))->SetObjectName("FEMeshFromBB_",Count);
    //vtkMimxMeshActor::SafeDownCast(this->FEMeshList->GetItem(
    //    this->FEMeshList->GetNumberOfItems()-1))->GetDataSet()->Modified();
    //this->GetMimxMainWindow()->GetRenderWidget()->AddViewProp(
    //    this->FEMeshList->GetItem(this->FEMeshList->GetNumberOfItems()-1)->GetActor());
    this->GetMimxMainWindow()->GetRenderWidget()->Render();
    this->GetMimxMainWindow()->GetRenderWidget()->ResetCamera();
          this->GetMimxMainWindow()->GetViewProperties()->AddObjectList(
                this->FEMeshList->GetItem(this->FEMeshList->GetNumberOfItems()-1));
          ugridfrombbox->RemoveObserver(callback);
          ugridfrombbox->Delete();
          equivalance->Delete();
        // creation of a new field data for element set storage
          ugrid = vtkMimxMeshActor::SafeDownCast(this->FEMeshList->GetItem(
                  this->FEMeshList->GetNumberOfItems()-1))->GetDataSet();

          vtkStringArray *elsetarray = vtkStringArray::New();
          elsetarray->SetName("Element_Set_Names");
          elsetarray->InsertNextValue(elementsetname);

          vtkStringArray *nodesetarray = vtkStringArray::New();
          nodesetarray->SetName("Node_Set_Names");
          nodesetarray->InsertNextValue(nodesetname);

          ugrid->GetFieldData()->AddArray(elsetarray);
          elsetarray->Delete();

          ugrid->GetFieldData()->AddArray(nodesetarray);
          nodesetarray->Delete();

          this->GetMimxMainWindow()->SetStatusText("Created Mesh");
          
          return 1;
  }
  ugridfrombbox->RemoveObserver(callback);
  ugridfrombbox->Delete();
        return 0;
}
//----------------------------------------------------------------------------
void vtkKWMimxCreateFEMeshFromBBGroup::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
//----------------------------------------------------------------------------
void vtkKWMimxCreateFEMeshFromBBGroup::CreateFEMeshFromBBCancelCallback()
{
        this->GetApplication()->Script("pack forget %s", this->MainFrame->GetWidgetName());
        this->MenuGroup->SetMenuButtonsEnabled(1);
          this->GetMimxMainWindow()->GetMainUserInterfacePanel()->GetMimxMainNotebook()->SetEnabled(1);
}
//------------------------------------------------------------------------------
void vtkKWMimxCreateFEMeshFromBBGroup::UpdateObjectLists()
{
  this->SurfaceListComboBox->GetWidget()->DeleteAllValues();
  
  if(this->OriginalPosition)
          this->OriginalPosition->Initialize();

  int defaultItem = -1;
  for (int i = 0; i < this->SurfaceList->GetNumberOfItems(); i++)
  {
    SurfaceListComboBox->GetWidget()->AddValue(
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
    SurfaceListComboBox->GetWidget()->SetValue(
          this->SurfaceList->GetItem(defaultItem)->GetFileName());
  }

  //for (i = 0; i < this->BBoxList->GetNumberOfItems(); i++)
  //{
         // vtkMimxUnstructuredGridActor *ugridactor = vtkMimxUnstructuredGridActor::
                //  SafeDownCast(this->BBoxList->GetItem(i));
         // vtkUnstructuredGrid *ugrid = ugridactor->GetDataSet();
         // if(ugrid->GetCellData()->GetArray("Mesh_Seed"))
         // {
                //  BBListComboBox->GetWidget()->AddValue(
                //        this->BBoxList->GetItem(i)->GetFileName());
                //  if(!this->OriginalPosition)
                //        this->OriginalPosition = vtkIntArray::New();
                //  this->OriginalPosition->InsertNextValue(i);
         // }
  //}
  
  this->BBListComboBox->GetWidget()->DeleteAllValues();
  
  defaultItem = -1;
        for (int i = 0; i < this->BBoxList->GetNumberOfItems(); i++)
        {
                vtkMimxUnstructuredGridActor *ugridactor = vtkMimxUnstructuredGridActor::
                      SafeDownCast(this->BBoxList->GetItem(i));
          vtkUnstructuredGrid *ugrid = ugridactor->GetDataSet();
          if (ugrid->GetCellData()->GetArray("Mesh_Seed"))
          {
                  BBListComboBox->GetWidget()->AddValue(
                          this->BBoxList->GetItem(i)->GetFileName());
                  if(!this->OriginalPosition)
                          this->OriginalPosition = vtkIntArray::New();
                  this->OriginalPosition->InsertNextValue(i);
                  
                  int viewedItem = this->GetMimxMainWindow()->GetRenderWidget()->GetRenderer()->HasViewProp(
                        this->BBoxList->GetItem(i)->GetActor());
      if ( (defaultItem == -1) && ( viewedItem ) )
                {
                  defaultItem = i;
                }
          }  
        }
        
        if (defaultItem != -1)
  {
    BBListComboBox->GetWidget()->SetValue(
          this->BBoxList->GetItem(defaultItem)->GetFileName());
  }
  
  /* This code does not work 
       this->DoUndoTree is always NULL
       
  if ( this->DoUndoTree != NULL )
  {
    defaultItem = -1;
    for (int i=0; i<this->DoUndoTree->GetNumberOfItems(); i++)
    {
          
          Node *currnode = this->DoUndoTree->GetItem(i);
*/
          /*while(currnode->Child != NULL)
          {
                currnode = currnode->Child;
          } 
          */
/*        BBListComboBox->GetWidget()->AddValue(
                          currnode->Data->GetFileName());
                          
                int viewedItem = this->GetMimxMainWindow()->GetRenderWidget()->GetRenderer()->HasViewProp(
                        currnode->Data->GetActor());
      if ( (defaultItem == -1) && ( viewedItem ) )
                {
                  defaultItem = i;
                }
    }
    
    if (defaultItem != -1)
    {
      BBListComboBox->GetWidget()->SetValue(
            this->DoUndoTree->GetItem(defaultItem)->Data->GetFileName());
    }
  }
*/
}
//------------------------------------------------------------------------------
void vtkKWMimxCreateFEMeshFromBBGroup::CreateFEMeshFromBBDoneCallback()
{
        if(this->CreateFEMeshFromBBApplyCallback())
                this->CreateFEMeshFromBBCancelCallback();
}
//---------------------------------------------------------------------------------
