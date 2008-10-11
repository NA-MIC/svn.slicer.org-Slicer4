/*=========================================================================

  Module:    $RCSfile: vtkMeshingWorkflowMRMLNotebook.cxx,v $

  Copyright (c) Kitware, Inc.
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkMeshingWorkflowMRMLNotebook.h"
//#include "vtkFESurfaceMRMLMenuGroup.h"
//#include "vtkFiniteElementMRMLMeshMenuGroup.h"
//#include "vtkMRMLScene.h"

#include "vtkActor.h"
#include "vtkPolyDataMapper.h"
#include "vtkSTLReader.h"

#include "vtkKWApplication.h"
#include "vtkKWFileBrowserDialog.h"
#include "vtkKWEvent.h"
#include "vtkKWFrame.h"
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
#include "vtkMath.h"
#include "vtkObjectFactory.h"

#include <vtksys/stl/list>
#include <vtksys/stl/algorithm>

#include "vtkKWMimxMainWindow.h"
#include  "vtkKWMimxBBMenuGroup.h"
#include  "vtkKWMimxImageMenuGroup.h"
#include  "vtkLinkedListWrapperTree.h"
#include  "vtkKWMimxQualityMenuGroup.h"
#include  "vtkKWMimxMaterialPropertyMenuGroup.h"


// define the option types
#define VTK_KW_OPTION_NONE         0
#define VTK_KW_OPTION_LOAD       1

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkMeshingWorkflowMRMLNotebook);
vtkCxxRevisionMacro(vtkMeshingWorkflowMRMLNotebook, "$Revision: 1.7 $");


//----------------------------------------------------------------------------
void vtkMeshingWorkflowMRMLNotebook::SetMimxMainWindow(vtkKWMimxMainWindow* mainwindow)
{
  this->MimxMainWindow = mainwindow;
}


//----------------------------------------------------------------------------
vtkKWMimxMainWindow* vtkMeshingWorkflowMRMLNotebook::GetMimxMainWindow(void)
{
  return MimxMainWindow;
}



//----------------------------------------------------------------------------
void vtkMeshingWorkflowMRMLNotebook::SetDoUndoTree(vtkLinkedListWrapperTree* doundotree)
{
  this->DoUndoTree = doundotree;
}


//----------------------------------------------------------------------------
vtkLinkedListWrapperTree* vtkMeshingWorkflowMRMLNotebook::GetDoUndoTree(void)
{
  return this->DoUndoTree;
}


//----------------------------------------------------------------------------
vtkMeshingWorkflowMRMLNotebook::vtkMeshingWorkflowMRMLNotebook()
{
  this->Notebook = NULL;
  this->BBMenuGroup = NULL;
  this->MimxViewWindow = NULL;
  this->SurfaceMenuGroup = NULL;
  this->FEMeshMenuGroup = NULL;
}

//----------------------------------------------------------------------------
vtkMeshingWorkflowMRMLNotebook::~vtkMeshingWorkflowMRMLNotebook()
{
  if(this->Notebook)
    this->Notebook->Delete();
  if(this->SurfaceMenuGroup)
    this->SurfaceMenuGroup->Delete();
  if(this->FEMeshMenuGroup)
    this->FEMeshMenuGroup->Delete();
}

// save reference to the scene to be used for storage 
 void vtkMeshingWorkflowMRMLNotebook::SetMRMLSceneForStorage(vtkMRMLScene* scene)
 {
    this->savedMRMLScene = scene;
 }

//----------------------------------------------------------------------------
void vtkMeshingWorkflowMRMLNotebook::CreateWidget()
{
  if(this->IsCreated())
  {
    vtkErrorMacro("class already created");
    return;
  }
  this->Superclass::CreateWidget();
  if(!this->Notebook)  this->Notebook = vtkKWNotebook::New();
  this->Notebook->SetParent(this);
  this->Notebook->Create();
  this->Notebook->UseFrameWithScrollbarsOn();
  this->GetApplication()->Script("pack %s -side top -anchor nw -expand y -padx 2 -pady 6", 
    this->Notebook->GetWidgetName());

  this->Notebook->AddPage("Image");
  this->Notebook->AddPage("Surface");
  this->Notebook->AddPage("Block(s)");
  this->Notebook->AddPage("Mesh");
  this->Notebook->AddPage("Quality");
  this->Notebook->AddPage("Materials");
 
  
  // instead of the local storage (commented out), instantiate a MRML-based list and pass the pointer to the 
  // current MRML scene to it, so it can manage the FE Surface Nodes in the current scene. 
  
  // ---------- Setup the FE Surface Tab in the Notebook -------
 
  //if(!this->SurfaceMenuGroup)  this->SurfaceMenuGroup = vtkFESurfaceMRMLMenuGroup::New();
  
//  if(this->savedMRMLScene)
//  {
//    cout << "setting MRML scene for Surface storage\n";
//    ((vtkFESurfaceMRMLMenuGroup*)this->SurfaceMenuGroup)->SetMRMLSceneForStorage(this->savedMRMLScene);
//  }
//  else 
//  {
//    cerr << "Tried to initialize Surface List with null MRML scene" << endl;
//  }
  
  if(!this->SurfaceMenuGroup)  this->SurfaceMenuGroup = vtkKWMimxSurfaceMenuGroup::New();
  this->SurfaceMenuGroup->SetParent(this->Notebook->GetFrame("Surface"));
  this->SurfaceMenuGroup->SetMimxMainWindow(this->GetMimxMainWindow());
  this->SurfaceMenuGroup->SetApplication(this->GetApplication());
  this->SurfaceMenuGroup->Create();
  this->GetApplication()->Script("pack %s -side top -anchor nw -expand y -padx 2 -pady 5", 
    this->SurfaceMenuGroup->GetWidgetName());

  if(!this->BBMenuGroup)  this->BBMenuGroup = vtkKWMimxBBMenuGroup::New();
   this->BBMenuGroup->SetParent(this->Notebook->GetFrame("Block(s)"));
   this->BBMenuGroup->SetMimxMainWindow(this->GetMimxMainWindow());
   this->BBMenuGroup->SetApplication(this->GetApplication());
   this->BBMenuGroup->SetDoUndoTree(this->DoUndoTree);
   this->BBMenuGroup->SetSurfaceMenuGroup(this->SurfaceMenuGroup);
   this->BBMenuGroup->Create();
   this->GetApplication()->Script("pack %s -side top -anchor nw -expand n -padx 2 -pady 5", 
           this->BBMenuGroup->GetWidgetName());

  // ---------- Setup the FE Surface Tab in the Notebook -------
  // generate menu items for FEMesh (which includes the mesh and bounding box)
  
  if(!this->FEMeshMenuGroup)  this->FEMeshMenuGroup = vtkKWMimxFEMeshMenuGroup::New();
  this->FEMeshMenuGroup->SetParent(this->Notebook->GetFrame("Mesh"));
  this->FEMeshMenuGroup->SetMimxMainWindow(this->GetMimxMainWindow());
  this->FEMeshMenuGroup->SetApplication(this->GetApplication());
  this->FEMeshMenuGroup->Create();
  this->GetApplication()->Script("pack %s -side top -anchor nw  -expand y -padx 2 -pady 5", 
  this->FEMeshMenuGroup->GetWidgetName());
  this->SetLists();

  // since we are using the MRML-based version, check for the existance of the MRML scene and attach
  // to it
  
//  if(this->savedMRMLScene)
//  {
//    cout << "setting MRML scene for BBox and Mesh storage\n";
//    ((vtkFiniteElementMRMLMeshMenuGroup*)this->FEMeshMenuGroup)->SetMRMLSceneForStorage(this->savedMRMLScene);
//  }
//  else 
//  {
//    cerr << "Tried to initialize Surface List with null MRML scene" << endl;
//  }
  
   //this->FEMeshMenuGroup->SetMimxViewWindow(this->GetMimxViewWindow());

 
}
//----------------------------------------------------------------------------
void vtkMeshingWorkflowMRMLNotebook::Update()
{
  this->UpdateEnableState();
}
//---------------------------------------------------------------------------
void vtkMeshingWorkflowMRMLNotebook::UpdateEnableState()
{
  this->Superclass::UpdateEnableState();
}
//----------------------------------------------------------------------------
void vtkMeshingWorkflowMRMLNotebook::SetLists()
{
  // list for surface operations
  this->SurfaceMenuGroup->SetFEMeshList(this->FEMeshMenuGroup->GetFEMeshList());
  this->SurfaceMenuGroup->SetBBoxList(this->BBMenuGroup->GetBBoxList());

  // list for BBlock operations
  this->BBMenuGroup->SetBBoxList(this->BBMenuGroup->GetBBoxList());
  // list for FEMesh operations
  this->FEMeshMenuGroup->SetSurfaceList(this->SurfaceMenuGroup->GetSurfaceList());
  this->FEMeshMenuGroup->SetBBoxList(this->BBMenuGroup->GetBBoxList());

}
//----------------------------------------------------------------------------
void vtkMeshingWorkflowMRMLNotebook::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
//----------------------------------------------------------------------------
