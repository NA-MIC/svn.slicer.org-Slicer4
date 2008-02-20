/*=========================================================================

  Module:    $RCSfile: vtkFESurfaceMRMLMenuGroup.cxx,v $

  Copyright (c) Kitware, Inc.
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkFESurfaceMRMLMenuGroup.h"

#include "vtkKWMimxSaveSTLSurfaceGroup.h"
#include "vtkKWMimxSaveVTKSurfaceGroup.h"

#include "vtkMimxActorBase.h"
#include "vtkMimxSurfacePolyDataActor.h"
#include "vtkKWMimxViewProperties.h"
#include "vtkKWMimxDeleteObjectGroup.h"

// declarations to interact with the MRML tree
#include "vtkFESurfaceList.h"
#include "vtkMRMLScene.h"

#include "vtkActor.h"
#include "vtkMath.h"
#include "vtkObjectFactory.h"
#include "vtkPolyDataMapper.h"
#include "vtkPolyDataReader.h"
#include "vtkSTLReader.h"

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

#include <vtksys/stl/list>
#include <vtksys/stl/algorithm>


//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkFESurfaceMRMLMenuGroup);
vtkCxxRevisionMacro(vtkFESurfaceMRMLMenuGroup, "$Revision: 1.8 $");

//----------------------------------------------------------------------------
vtkFESurfaceMRMLMenuGroup::vtkFESurfaceMRMLMenuGroup()
{
  this->ObjectMenuButton = vtkKWMenuButtonWithLabel::New();
  this->OperationMenuButton = NULL;
  this->TypeMenuButton = NULL;
//  this->SurfaceList = vtkLinkedListWrapper::New();
  this->SurfaceList = vtkFESurfaceList::New();
  this->MimxViewProperties = NULL;
  this->FileBrowserDialog = NULL;
  this->SaveSTLGroup = NULL;
  this->SaveVTKGroup = NULL;
  this->DeleteObjectGroup = NULL;
}

//----------------------------------------------------------------------------
vtkFESurfaceMRMLMenuGroup::~vtkFESurfaceMRMLMenuGroup()
{
  if (this->SurfaceList) {
    for (int i=0; i < this->SurfaceList->GetNumberOfItems(); i++) {
      vtkMimxSurfacePolyDataActor::SafeDownCast(
        this->SurfaceList->GetItem(i))->Delete();
    }
    this->SurfaceList->Delete();
  }
  if(this->ObjectMenuButton)
    this->ObjectMenuButton->Delete();
  if(this->OperationMenuButton)
    this->OperationMenuButton->Delete();
  if(this->TypeMenuButton)
    this->TypeMenuButton->Delete();
  if(this->MimxViewProperties)
    this->MimxViewProperties->Delete();
  if(this->FileBrowserDialog)
    this->FileBrowserDialog->Delete();
  if(this->SaveSTLGroup)
  this->SaveSTLGroup->Delete();
  if(this->SaveVTKGroup)
    this->SaveVTKGroup->Delete();
  if(this->DeleteObjectGroup)
    this->DeleteObjectGroup->Delete();
}


// save reference to the scene to be used for storage.  Pass the reference onto the list.
// We are using a cast here because the parent for all lists doesn't have the MRML scene method
// defined. 

 void vtkFESurfaceMRMLMenuGroup::SetMRMLSceneForStorage(vtkMRMLScene* scene)
 {
    this->savedMRMLScene = scene; 
    ((vtkFESurfaceList*)this->SurfaceList)->SetMRMLSceneForStorage(scene);
 }
 
 
//----------------------------------------------------------------------------
void vtkFESurfaceMRMLMenuGroup::CreateWidget()
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
  this->MainFrame->SetLabelText("Surface Operations");

  this->GetApplication()->Script(
    "pack %s -side top -anchor nw -expand y -padx 2 -pady 6", 
    this->MainFrame->GetWidgetName());

  this->ObjectMenuButton->SetParent(this->MainFrame->GetFrame());
  this->ObjectMenuButton->Create();
  this->ObjectMenuButton->SetBorderWidth(2);
  this->ObjectMenuButton->SetReliefToGroove();
  this->ObjectMenuButton->SetLabelText("Object :");
  this->ObjectMenuButton->SetPadX(2);
  this->ObjectMenuButton->SetPadY(2);
  this->GetApplication()->Script("pack %s -side top -anchor nw -expand y -padx 2 -pady 5", 
    this->ObjectMenuButton->GetWidgetName());
  this->ObjectMenuButton->GetWidget()->GetMenu()->AddRadioButton(
    "Surface",this, "SurfaceMenuCallback");
  // declare operations menu
  if(!this->OperationMenuButton)  
    this->OperationMenuButton = vtkKWMenuButtonWithLabel::New();
  this->OperationMenuButton->SetParent(this->MainFrame->GetFrame());
  this->OperationMenuButton->Create();
  this->OperationMenuButton->SetBorderWidth(2);
  this->OperationMenuButton->SetReliefToGroove();
  this->OperationMenuButton->SetLabelText("Operation :");
  this->OperationMenuButton->SetPadX(2);
  this->OperationMenuButton->SetPadY(2);
  this->GetApplication()->Script("pack %s -side top -anchor nw -expand y -padx 2 -pady 5", 
    this->OperationMenuButton->GetWidgetName());
  this->OperationMenuButton->SetEnabled(0);
  // declare operations menu
  if(!this->TypeMenuButton)  
    this->TypeMenuButton = vtkKWMenuButtonWithLabel::New();
  this->TypeMenuButton->SetParent(this->MainFrame->GetFrame());
  this->TypeMenuButton->Create();
  this->TypeMenuButton->SetBorderWidth(2);
  this->TypeMenuButton->SetReliefToGroove();
  this->TypeMenuButton->SetLabelText("Type :");
  this->TypeMenuButton->SetPadX(2);
  this->TypeMenuButton->SetPadY(2);
  this->GetApplication()->Script("pack %s -side top -anchor nw -expand y -padx 2 -pady 5", 
    this->TypeMenuButton->GetWidgetName());
  this->TypeMenuButton->SetEnabled(0);
}
//----------------------------------------------------------------------------
void vtkFESurfaceMRMLMenuGroup::Update()
{
  this->UpdateEnableState();
}
//----------------------------------------------------------------------------
void vtkFESurfaceMRMLMenuGroup::UpdateEnableState()
{
  this->Superclass::UpdateEnableState();
}
//----------------------------------------------------------------------------
void vtkFESurfaceMRMLMenuGroup::SurfaceMenuCallback()
{
  this->OperationMenuButton->GetWidget()->GetMenu()->DeleteAllItems();
  this->OperationMenuButton->GetWidget()->GetMenu()->AddRadioButton(
    "Load",this, "LoadSurfaceCallback");

  this->OperationMenuButton->GetWidget()->GetMenu()->AddRadioButton(
    "Save",this, "SaveSurfaceCallback");

  this->OperationMenuButton->GetWidget()->GetMenu()->AddRadioButton(
    "Delete",this, "DeleteSurfaceCallback");

  this->OperationMenuButton->SetEnabled(1);
  // view properties menu item
  if(!this->MimxViewProperties)
  {
    this->MimxViewProperties = vtkKWMimxViewProperties::New();
  }
  this->MimxViewProperties->SetParent(this->GetParent());
  this->MimxViewProperties->SetObjectList(this->SurfaceList);
  this->MimxViewProperties->SetMimxViewWindow(this->GetMimxViewWindow());
  this->MimxViewProperties->Create();
  this->GetApplication()->Script("pack %s -side top -anchor nw -expand y -padx 2 -pady 5", 
    this->MimxViewProperties->GetMainFrame()->GetWidgetName());
}
//----------------------------------------------------------------------------
void vtkFESurfaceMRMLMenuGroup::LoadSurfaceCallback()
{
  this->TypeMenuButton->GetWidget()->GetMenu()->DeleteAllItems();
  this->TypeMenuButton->GetWidget()->GetMenu()->AddRadioButton(
    "STL",this, "LoadSTLSurfaceCallback");
  this->TypeMenuButton->GetWidget()->GetMenu()->AddRadioButton(
    "VTK",this, "LoadVTKSurfaceCallback");
  this->TypeMenuButton->SetEnabled(1);
}
//----------------------------------------------------------------------------
void vtkFESurfaceMRMLMenuGroup::SaveSurfaceCallback()
{
  this->TypeMenuButton->GetWidget()->GetMenu()->DeleteAllItems();
  this->TypeMenuButton->GetWidget()->GetMenu()->AddRadioButton(
    "STL",this, "SaveSTLSurfaceCallback");
  this->TypeMenuButton->GetWidget()->GetMenu()->AddRadioButton(
    "VTK",this, "SaveVTKSurfaceCallback");
  this->TypeMenuButton->GetWidget()->SetValue("");
  this->TypeMenuButton->SetEnabled(1);
}
//----------------------------------------------------------------------------
void vtkFESurfaceMRMLMenuGroup::LoadSTLSurfaceCallback()
{
  if(!this->FileBrowserDialog)
  {
    this->FileBrowserDialog = vtkKWFileBrowserDialog::New();
    this->FileBrowserDialog->SetApplication(this->GetApplication());
    this->FileBrowserDialog->Create();
  }
  this->FileBrowserDialog->SetDefaultExtension(".stl");
  this->FileBrowserDialog->SetFileTypes("{{STL files} {.stl}}");
  this->FileBrowserDialog->Invoke();
  

  if(this->FileBrowserDialog->GetStatus() == vtkKWDialog::StatusOK)
  {
    if(this->FileBrowserDialog->GetFileName())
    {
      const char *filename = this->FileBrowserDialog->GetFileName();
      cout << "*** surface filename was: " << filename << endl;
      vtkSTLReader *reader = vtkSTLReader::New();
      reader->SetFileName(this->FileBrowserDialog->GetFileName());
      reader->Update();
      cout << "*** Surface MRML menu reader completed successfully" << endl;
      if(reader->GetOutput())
      {

        
//        this->SurfaceList->AppendItem(vtkMimxSurfacePolyDataActor::New());
//        ((vtkFESurfaceList*)this->SurfaceList)->AppendItem(actor);
//  
//        this->SurfaceList->GetItem(this->SurfaceList->GetNumberOfItems()-1)->SetFilePath(
//          this->FileBrowserDialog->GetFileName());
// 
//        this->SurfaceList->GetItem(this->SurfaceList->GetNumberOfItems()-1)->SetFileName(
//          this->ExtractFileName(this->FileBrowserDialog->GetFileName()));
////        vtkMimxSurfacePolyDataActor *actor = NULL;
////        this->SurfaceList->GetItem(this->SurfaceList->GetNumberOfItems()-1);
//        vtkMimxSurfacePolyDataActor::SafeDownCast(this->SurfaceList->GetItem(
//          this->SurfaceList->GetNumberOfItems()-1))->GetDataSet()->DeepCopy(reader->GetOutput());
// 
//        vtkMimxSurfacePolyDataActor::SafeDownCast(this->SurfaceList->GetItem(
//        this->SurfaceList->GetNumberOfItems()-1))->GetDataSet()->Modified();         
//        this->GetMimxViewWindow()->GetRenderWidget()->AddViewProp(
//        this->SurfaceList->GetItem(this->SurfaceList->GetNumberOfItems()-1)->GetActor());
//        this->GetMimxViewWindow()->GetRenderWidget()->Render();
//        this->GetMimxViewWindow()->GetRenderWidget()->ResetCamera();
//        this->MimxViewProperties->AddObjectList(); 
        
    // *** original code created the entry, then retreived it and filled it.  Our initial solution
    // will be to fill the records before adding them to MRML, since this avoids the 
    // complexity of adding observers to the MimxActor records.  A more robust solution needs
    // to be considered. 

        vtkMimxSurfacePolyDataActor *actor = vtkMimxSurfacePolyDataActor::New();
        vtkFESurfaceList *surflist = (vtkFESurfaceList*)(this->SurfaceList);

        // set the filename and path in the actor, then copy the dataset from the reader
        actor->SetFilePath(this->FileBrowserDialog->GetFileName());
        // defeat extractfilename for now, as the function doesn't work.
        actor->SetFileName(this->ExtractFileName(this->FileBrowserDialog->GetFileName()));
        //actor->SetFileName(this->FileBrowserDialog->GetFileName());
        
        // copy the dataset over to MRML node
        vtkMimxSurfacePolyDataActor::SafeDownCast(actor)->GetDataSet()->DeepCopy(reader->GetOutput());
        vtkMimxSurfacePolyDataActor::SafeDownCast(actor)->GetDataSet()->Modified();       

        // now add this to the MRML tree
        surflist->AppendItem(actor);
        
        // add actor to the slicer viewer (unnecessary because of presence in MRML?) and force a redraw
        this->GetMimxViewWindow()->GetRenderWidget()->AddViewProp(actor->GetActor());
        this->GetMimxViewWindow()->GetRenderWidget()->Render();
        //this->GetMimxViewWindow()->GetRenderWidget()->ResetCamera();
        this->MimxViewProperties->AddObjectList(); 

      }
      reader->Delete();
    }
  }
}
//----------------------------------------------------------------------------
void vtkFESurfaceMRMLMenuGroup::LoadVTKSurfaceCallback()
{
  if(!this->FileBrowserDialog)
  {
    this->FileBrowserDialog = vtkKWFileBrowserDialog::New();
    this->FileBrowserDialog->SetApplication(this->GetApplication());
    this->FileBrowserDialog->Create();
  }
  this->FileBrowserDialog->SetDefaultExtension(".vtk");
  this->FileBrowserDialog->SetFileTypes("{{VTK polydata files} {.vtk}}");
  this->FileBrowserDialog->Invoke();
  if(this->FileBrowserDialog->GetStatus() == vtkKWDialog::StatusOK)
  {
    if(this->FileBrowserDialog->GetFileName())
    {
      const char *filename = this->FileBrowserDialog->GetFileName();
      vtkPolyDataReader *reader = vtkPolyDataReader::New();
      reader->SetFileName(this->FileBrowserDialog->GetFileName());
      reader->Update();
      if(reader->GetOutput())
      {
////        vtkMimxSurfacePolyDataActor *actor = vtkMimxSurfacePolyDataActor::New();
//        this->SurfaceList->AppendItem(vtkMimxSurfacePolyDataActor::New());
//        this->SurfaceList->GetItem(this->SurfaceList->GetNumberOfItems()-1)->SetFilePath(
//          this->FileBrowserDialog->GetFileName());
//        this->SurfaceList->GetItem(this->SurfaceList->GetNumberOfItems()-1)->SetFileName(
//          this->ExtractFileName(this->FileBrowserDialog->GetFileName()));
////        vtkMimxSurfacePolyDataActor *actor = NULL;
////        this->SurfaceList->GetItem(this->SurfaceList->GetNumberOfItems()-1);
//        vtkMimxSurfacePolyDataActor::SafeDownCast(this->SurfaceList->GetItem(
//          this->SurfaceList->GetNumberOfItems()-1))->GetDataSet()->DeepCopy(reader->GetOutput());
//        vtkMimxSurfacePolyDataActor::SafeDownCast(this->SurfaceList->GetItem(
//          this->SurfaceList->GetNumberOfItems()-1))->GetDataSet()->Modified();
//        this->GetMimxViewWindow()->GetRenderWidget()->AddViewProp(
//          this->SurfaceList->GetItem(this->SurfaceList->GetNumberOfItems()-1)->GetActor());
//        this->GetMimxViewWindow()->GetRenderWidget()->Render();
//        this->GetMimxViewWindow()->GetRenderWidget()->ResetCamera();
//        this->MimxViewProperties->AddObjectList();
        

  // *** original code created the entry, then retreived it and filled it.  Our initial solution
  // will be to fill the records before adding them to MRML, since this avoids the 
  // complexity of adding observers to the MimxActor records.  A more robust solution needs
  // to be considered. 


          vtkFESurfaceList *surflist = (vtkFESurfaceList*)(this->SurfaceList);

          surflist->GetItem(this->SurfaceList->GetNumberOfItems()-1)->SetFilePath(
                 this->FileBrowserDialog->GetFileName());
          
          surflist->GetItem(surflist->GetNumberOfItems()-1)->SetFileName(
                  this->ExtractFileName(this->FileBrowserDialog->GetFileName()));
          vtkMimxSurfacePolyDataActor::SafeDownCast(surflist->GetItem(
                surflist->GetNumberOfItems()-1))->GetDataSet()->DeepCopy(reader->GetOutput());
          vtkMimxSurfacePolyDataActor::SafeDownCast(surflist->GetItem(
             surflist->GetNumberOfItems()-1))->GetDataSet()->Modified();
       
//          this->GetMimxViewWindow()->GetRenderWidget()->AddViewProp(
//            surflist->GetItem(surflist->GetNumberOfItems()-1)->GetActor());
//          surflist->AppendItem(vtkMimxSurfacePolyDataActor::New());
//          this->GetMimxViewWindow()->GetRenderWidget()->Render();
//          this->GetMimxViewWindow()->GetRenderWidget()->ResetCamera();
//          this->MimxViewProperties->AddObjectList();
          
      }
      reader->Delete();
    }
  }
}
//----------------------------------------------------------------------------
void vtkFESurfaceMRMLMenuGroup::SaveSTLSurfaceCallback()
{
  if(this->SaveSTLGroup)
  {
    this->SaveSTLGroup->Delete();
  }
  this->SaveSTLGroup = vtkKWMimxSaveSTLSurfaceGroup::New();
  this->SaveSTLGroup->SetParent(this->MainFrame->GetFrame());

  // downcast to use the MRML-based list
  vtkFESurfaceList *surflist = (vtkFESurfaceList*)(this->SurfaceList);

  this->SaveSTLGroup->SetSurfaceList(surflist);
  this->SaveSTLGroup->SetMenuGroup(this);
  this->SaveSTLGroup->Create();
  this->SetMenuButtonsEnabled(0);
  this->GetApplication()->Script("pack %s -side top -anchor nw -expand y -padx 2 -pady 5", 
    this->SaveSTLGroup->GetMainFrame()->GetWidgetName());
}
//----------------------------------------------------------------------------
void vtkFESurfaceMRMLMenuGroup::SaveVTKSurfaceCallback()
{
  if(this->SaveVTKGroup)
  {
    this->SaveVTKGroup->Delete();
  }
    this->SaveVTKGroup = vtkKWMimxSaveVTKSurfaceGroup::New();
    this->SaveVTKGroup->SetParent(this->MainFrame->GetFrame());
    // downcast to use the MRML-based list
    vtkFESurfaceList *surflist = (vtkFESurfaceList*)(this->SurfaceList);
    this->SaveVTKGroup->SetSurfaceList(surflist);
    
    this->SaveVTKGroup->SetMenuGroup(this);
    this->SaveVTKGroup->Create();

  this->SetMenuButtonsEnabled(0);
  this->GetApplication()->Script("pack %s -side top -anchor nw -expand y -padx 2 -pady 5", 
    this->SaveVTKGroup->GetMainFrame()->GetWidgetName());
}
//----------------------------------------------------------------------------
void vtkFESurfaceMRMLMenuGroup::DeleteSurfaceCallback()
{
  if(this->DeleteObjectGroup)
  {
    this->DeleteObjectGroup->Delete();
  }
  this->DeleteObjectGroup = vtkKWMimxDeleteObjectGroup::New();
  this->DeleteObjectGroup->SetParent(this->MainFrame->GetFrame());
  vtkFESurfaceList *surflist = (vtkFESurfaceList*)(this->SurfaceList);
  this->DeleteObjectGroup->SetSurfaceList(surflist);
  this->DeleteObjectGroup->SetMenuGroup(this);
  this->DeleteObjectGroup->SetViewProperties(this->MimxViewProperties);
  this->DeleteObjectGroup->Create();
  this->TypeMenuButton->GetWidget()->SetValue("");
  this->TypeMenuButton->GetWidget()->GetMenu()->DeleteAllItems();
  this->TypeMenuButton->SetEnabled(0);
  this->SetMenuButtonsEnabled(0);
  this->DeleteObjectGroup->GetMainFrame()->SetLabelText("Delete Surfaces");
  this->GetApplication()->Script("pack %s -side top -anchor nw -expand y -padx 2 -pady 5", 
    this->DeleteObjectGroup->GetMainFrame()->GetWidgetName());
}
//----------------------------------------------------------------------------
void vtkFESurfaceMRMLMenuGroup::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
//---------------------------------------------------------------------------
