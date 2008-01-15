/*=========================================================================

  Module:    $RCSfile: vtkFiniteElementCreateFEMeshFromBBGroup.cxx,v $

  Copyright (c) Kitware, Inc.
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkFiniteElementCreateFEMeshFromBBGroup.h"
#include "vtkKWMimxViewWindow.h"

#include "vtkLinkedListWrapper.h"
#include "vtkFiniteElementMeshList.h"
#include "vtkFESurfaceList.h"
#include "vtkFiniteElementBoundingBoxList.h"
#include "vtkFiniteElementMeshList.h"

#include "vtkActor.h"
#include "vtkCellData.h"
#include "vtkMimxSurfacePolyDataActor.h"
#include "vtkPolyData.h"
#include "vtkProperty.h"
#include "vtkMimxUnstructuredGridActor.h"
#include "vtkUnstructuredGrid.h"
#include "vtkMimxUnstructuredGridFromBoundingBox.h"

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

#include <vtksys/stl/list>
#include <vtksys/stl/algorithm>

// define the option types
#define VTK_KW_OPTION_NONE         0
#define VTK_KW_OPTION_LOAD       1

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkFiniteElementCreateFEMeshFromBBGroup);
vtkCxxRevisionMacro(vtkFiniteElementCreateFEMeshFromBBGroup, "$Revision: 1.3 $");

//----------------------------------------------------------------------------
vtkFiniteElementCreateFEMeshFromBBGroup::vtkFiniteElementCreateFEMeshFromBBGroup()
{
  this->SurfaceListComboBox = NULL;
  this->BBListComboBox = NULL;
}

//----------------------------------------------------------------------------
vtkFiniteElementCreateFEMeshFromBBGroup::~vtkFiniteElementCreateFEMeshFromBBGroup()
{
  if(this->SurfaceListComboBox)
     this->SurfaceListComboBox->Delete();
  if(this->BBListComboBox)
    this->BBListComboBox->Delete();
 }
//----------------------------------------------------------------------------
void vtkFiniteElementCreateFEMeshFromBBGroup::CreateWidget()
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
  this->MainFrame->Create();
  this->MainFrame->SetLabelText("Create FE Mesh From Bounds");

  this->GetApplication()->Script(
    "pack %s -side top -anchor nw -expand y -padx 2 -pady 6", 
    this->MainFrame->GetWidgetName());

  SurfaceListComboBox->SetParent(this->MainFrame->GetFrame());
  SurfaceListComboBox->Create();
  SurfaceListComboBox->SetLabelText("Surface Selection : ");
  SurfaceListComboBox->GetWidget()->ReadOnlyOn();
  SurfaceListComboBox->GetWidget()->SetBalloonHelpString("Surface onto which the resulting FE Mesh projected");
  int i;
  for (i = 0; i < this->SurfaceList->GetNumberOfItems(); i++)
  {
    SurfaceListComboBox->GetWidget()->AddValue(
      this->SurfaceList->GetItem(i)->GetFileName());
  }

  this->GetApplication()->Script(
    "pack %s -side top -anchor nw -expand y -padx 2 -pady 6", 
    SurfaceListComboBox->GetWidgetName());

  BBListComboBox->SetParent(this->MainFrame->GetFrame());
  BBListComboBox->Create();
  BBListComboBox->SetLabelText("Bounding box Selection : ");
  BBListComboBox->GetWidget()->ReadOnlyOn();
  BBListComboBox->GetWidget()->SetBalloonHelpString("Bounding box from which F E Mesh is generated");

  for (i = 0; i < this->BBoxList->GetNumberOfItems(); i++)
  {
    BBListComboBox->GetWidget()->AddValue(
      this->BBoxList->GetItem(i)->GetFileName());
  }

  this->GetApplication()->Script(
    "pack %s -side top -anchor nw -expand y -padx 2 -pady 6", 
    BBListComboBox->GetWidgetName());

  this->DoneButton->SetParent(this->MainFrame->GetFrame());
  this->DoneButton->Create();
  this->DoneButton->SetText("Done");
  this->DoneButton->SetCommand(this, "CreateFEMeshFromBBCallback");
  this->GetApplication()->Script(
    "pack %s -side left -anchor nw -expand y -padx 20 -pady 6", 
    this->DoneButton->GetWidgetName());

  this->CancelButton->SetParent(this->MainFrame->GetFrame());
  this->CancelButton->Create();
  this->CancelButton->SetText("Cancel");
  this->CancelButton->SetCommand(this, "CreateFEMeshFromBBCancelCallback");
  this->GetApplication()->Script(
    "pack %s -side left -anchor nw -expand y -padx 2 -pady 6", 
    this->CancelButton->GetWidgetName());

}
//----------------------------------------------------------------------------
void vtkFiniteElementCreateFEMeshFromBBGroup::Update()
{
  this->UpdateEnableState();
}
//---------------------------------------------------------------------------
void vtkFiniteElementCreateFEMeshFromBBGroup::UpdateEnableState()
{
  this->Superclass::UpdateEnableState();
}
//----------------------------------------------------------------------------
void vtkFiniteElementCreateFEMeshFromBBGroup::CreateFEMeshFromBBCallback()
{
  if(strcmp(this->SurfaceListComboBox->GetWidget()->GetValue(),"") && 
    strcmp(this->BBListComboBox->GetWidget()->GetValue(),""))
  {
    vtkKWComboBox *combobox = this->SurfaceListComboBox->GetWidget();
    const char *name = combobox->GetValue();
 
    // subcast to make sure we are using the MRML-based storage for surface.
    // Get the polydata for the surface by extracting from the MRML tree element
    vtkFESurfaceList *surflist = (vtkFESurfaceList*)(this->SurfaceList);
    vtkPolyData *polydata = vtkMimxSurfacePolyDataActor::SafeDownCast(
            surflist->GetItem(combobox->GetValueIndex(name)))->GetDataSet();

    // get the bbox name, then get a pointer to the selected bbox
    combobox = this->BBListComboBox->GetWidget();
    name = combobox->GetValue();

    // subclass to use MRML-based storage for the bbox list
    vtkFiniteElementBoundingBoxList *bblist = (vtkFiniteElementBoundingBoxList*)(this->BBoxList);
    
    // get the unstructured grid from the current bbox
        vtkMimxUnstructuredGridActor *ugridactor = vtkMimxUnstructuredGridActor::
          SafeDownCast(bblist->GetItem(combobox->GetValueIndex(name)));
        vtkUnstructuredGrid *ugrid = ugridactor->GetDataSet();
    
    if(!ugrid->GetCellData()->GetScalars("Mesh_Seed"))
    {
        vtkErrorMacro("MRML didn't find Mesh_Seed scalars");
        return;
    }

    vtkMimxUnstructuredGridFromBoundingBox *ugridfrombbox = 
      vtkMimxUnstructuredGridFromBoundingBox::New();
       
       // run algorithm to calculate mesh from the surface and bbox
       cout << "Hooray! Running Mesh from bbox algorithm" << endl;
       ugridfrombbox->SetBoundingBox(ugrid);
       ugridfrombbox->SetSurface(polydata);
       ugridfrombbox->Update();
    
    if (ugridfrombbox->GetOutput())
    {
//      this->FEMeshList->AppendItem(vtkMimxUnstructuredGridActor::New());
//      vtkMimxUnstructuredGridActor::SafeDownCast(this->FEMeshList->GetItem(
//        this->FEMeshList->GetNumberOfItems()-1))->GetDataSet()->
//        DeepCopy(ugridfrombbox->GetOutput());
//      this->Count++;
//      vtkMimxUnstructuredGridActor::SafeDownCast(this->FEMeshList->GetItem(
//        this->FEMeshList->GetNumberOfItems()-1))->SetObjectName("FEMeshFromBB_",Count);
//      vtkMimxUnstructuredGridActor::SafeDownCast(this->FEMeshList->GetItem(
//        this->FEMeshList->GetNumberOfItems()-1))->GetDataSet()->Modified();
//    this->ViewProperties->AddObjectList();
//      this->GetMimxViewWindow()->GetRenderWidget()->AddViewProp(
//        this->FEMeshList->GetItem(this->FEMeshList->GetNumberOfItems()-1)->GetActor());
//      this->GetMimxViewWindow()->GetRenderWidget()->Render();
//      this->GetMimxViewWindow()->GetRenderWidget()->ResetCamera();

        // subcast to make sure we are using the MRML-based storage for the mesh list
       vtkFiniteElementMeshList *meshlist = (vtkFiniteElementMeshList*)(this->FEMeshList);
 
         cout << "MeshList MRML node about to be created" << endl;       
         meshlist->AppendItem(vtkMimxUnstructuredGridActor::New());
       vtkMimxUnstructuredGridActor::SafeDownCast(meshlist->GetItem(
         meshlist->GetNumberOfItems()-1))->GetDataSet()->
         DeepCopy(ugridfrombbox->GetOutput());
       this->Count++;
       vtkMimxUnstructuredGridActor::SafeDownCast(meshlist->GetItem(
         meshlist->GetNumberOfItems()-1))->SetObjectName("FEMeshFromBB_",Count);
       vtkMimxUnstructuredGridActor::SafeDownCast(meshlist->GetItem(
         meshlist->GetNumberOfItems()-1))->GetDataSet()->Modified();
       this->ViewProperties->AddObjectList();
 
       // *** change
       // Add the actor to the window;  this should observe the MRML instead
       this->GetMimxViewWindow()->GetRenderWidget()->AddViewProp(
         meshlist->GetItem(meshlist->GetNumberOfItems()-1)->GetActor());      
       this->GetMimxViewWindow()->GetRenderWidget()->Render();
       this->GetMimxViewWindow()->GetRenderWidget()->ResetCamera();
      
    }
    else {
      vtkErrorMacro("*** No UGrid output from Bounding Box");
    }
    ugridfrombbox->Delete();
  }
}
//----------------------------------------------------------------------------
void vtkFiniteElementCreateFEMeshFromBBGroup::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
//----------------------------------------------------------------------------
void vtkFiniteElementCreateFEMeshFromBBGroup::CreateFEMeshFromBBCancelCallback()
{
  this->GetApplication()->Script("pack forget %s", this->MainFrame->GetWidgetName());
  this->MenuGroup->SetMenuButtonsEnabled(1);
}
//------------------------------------------------------------------------------
void vtkFiniteElementCreateFEMeshFromBBGroup::UpdateObjectLists()
{
  this->SurfaceListComboBox->GetWidget()->DeleteAllValues();
  this->BBListComboBox->GetWidget()->DeleteAllValues();

  int i;
  for (i = 0; i < this->SurfaceList->GetNumberOfItems(); i++)
  {
    SurfaceListComboBox->GetWidget()->AddValue(
      this->SurfaceList->GetItem(i)->GetFileName());
  }

  for (i = 0; i < this->BBoxList->GetNumberOfItems(); i++)
  {
    BBListComboBox->GetWidget()->AddValue(
      this->BBoxList->GetItem(i)->GetFileName());
  }
}
//------------------------------------------------------------------------------
