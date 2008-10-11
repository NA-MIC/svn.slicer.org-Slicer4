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
#include "vtkFESurfaceMRMLMenuGroup.h"
#include "vtkFiniteElementMRMLMeshMenuGroup.h"
#include "vtkMRMLScene.h"

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

// define the option types
#define VTK_KW_OPTION_NONE         0
#define VTK_KW_OPTION_LOAD       1

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkMeshingWorkflowMRMLNotebook);
vtkCxxRevisionMacro(vtkMeshingWorkflowMRMLNotebook, "$Revision: 1.7 $");

//----------------------------------------------------------------------------
vtkMeshingWorkflowMRMLNotebook::vtkMeshingWorkflowMRMLNotebook()
{
  this->Notebook = NULL;
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
  this->Notebook->AddPage("F E Mesh");
  this->Notebook->AddPage("Mesh Quality");
  
  // instead of the local storage (commented out), instantiate a MRML-based list and pass the pointer to the 
  // current MRML scene to it, so it can manage the FE Surface Nodes in the current scene. 
  
  // ---------- Setup the FE Surface Tab in the Notebook -------
  //  if(!this->SurfaceMenuGroup)  this->SurfaceMenuGroup = vtkKWMimxSurfaceMenuGroup::New();
  if(!this->SurfaceMenuGroup)  this->SurfaceMenuGroup = vtkFESurfaceMRMLMenuGroup::New();
  
  if(this->savedMRMLScene)
  {
    cout << "setting MRML scene for Surface storage\n";
    ((vtkFESurfaceMRMLMenuGroup*)this->SurfaceMenuGroup)->SetMRMLSceneForStorage(this->savedMRMLScene);
  }
  else 
  {
    cerr << "Tried to initialize Surface List with null MRML scene" << endl;
  }
  
  this->SurfaceMenuGroup->SetParent(this->Notebook->GetFrame("Surface"));
  this->SurfaceMenuGroup->SetMimxViewWindow(this->GetMimxViewWindow());
  this->SurfaceMenuGroup->SetApplication(this->GetApplication());

  this->SurfaceMenuGroup->Create();

  this->GetApplication()->Script("pack %s -side top -anchor nw -expand y -padx 2 -pady 5", 
    this->SurfaceMenuGroup->GetWidgetName());

  // ---------- Setup the FE Surface Tab in the Notebook -------
  // generate menu items for FEMesh (which includes the mesh and bounding box)
  
//  if(!this->FEMeshMenuGroup)  this->FEMeshMenuGroup = vtkKWMimxFEMeshMenuGroup::New();
  if(!this->FEMeshMenuGroup)  this->FEMeshMenuGroup = vtkFiniteElementMRMLMeshMenuGroup::New();

  this->FEMeshMenuGroup->SetParent(this->Notebook->GetFrame("F E Mesh"));
  this->FEMeshMenuGroup->SetMimxViewWindow(this->GetMimxViewWindow());
  this->FEMeshMenuGroup->SetApplication(this->GetApplication());

  // since we are using the MRML-based version, check for the existance of the MRML scene and attach
  // to it
  
  if(this->savedMRMLScene)
  {
    cout << "setting MRML scene for BBox and Mesh storage\n";
    ((vtkFiniteElementMRMLMeshMenuGroup*)this->FEMeshMenuGroup)->SetMRMLSceneForStorage(this->savedMRMLScene);
  }
  else 
  {
    cerr << "Tried to initialize Surface List with null MRML scene" << endl;
  }
  
  this->FEMeshMenuGroup->Create();
  this->FEMeshMenuGroup->SetMimxViewWindow(this->GetMimxViewWindow());

  this->GetApplication()->Script("pack %s -side top -anchor nw  -expand y -padx 2 -pady 5", 
    this->FEMeshMenuGroup->GetWidgetName());
  this->SetLists();
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
  // list for FEMesh operations
  this->FEMeshMenuGroup->SetSurfaceList(this->SurfaceMenuGroup->GetSurfaceList());
}
//----------------------------------------------------------------------------
void vtkMeshingWorkflowMRMLNotebook::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
//----------------------------------------------------------------------------
