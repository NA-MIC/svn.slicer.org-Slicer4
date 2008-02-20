/*=========================================================================

  Module:    $RCSfile: vtkFiniteElementMRMLCreateBBFromBoundsGroup.cxx,v $

  Copyright (c) Kitware, Inc.
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkFiniteElementMRMLCreateBBFromBoundsGroup.h"
#include "vtkKWMimxViewWindow.h"
#include "vtkKWMimxMainMenuGroup.h"

#include "vtkLinkedListWrapper.h"

// added for MRML integration
#include "vtkFiniteElementBoundingBoxList.h"
#include "vtkMimxSurfacePolyDataActor.h"

#include "vtkActor.h"
#include "vtkMimxBoundingBoxSource.h"
#include "vtkMimxUnstructuredGridActor.h"
#include "vtkPolyData.h"
#include "vtkProperty.h"
#include "vtkMimxUnstructuredGridActor.h"
#include "vtkUnstructuredGrid.h"

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
vtkStandardNewMacro(vtkFiniteElementMRMLCreateBBFromBoundsGroup);
vtkCxxRevisionMacro(vtkFiniteElementMRMLCreateBBFromBoundsGroup, "$Revision: 1.5 $");

//----------------------------------------------------------------------------
vtkFiniteElementMRMLCreateBBFromBoundsGroup::vtkFiniteElementMRMLCreateBBFromBoundsGroup()
{
  this->ObjectListComboBox = NULL;
}

//----------------------------------------------------------------------------
vtkFiniteElementMRMLCreateBBFromBoundsGroup::~vtkFiniteElementMRMLCreateBBFromBoundsGroup()
{
  if(this->ObjectListComboBox)
     this->ObjectListComboBox->Delete();
 }
//----------------------------------------------------------------------------
void vtkFiniteElementMRMLCreateBBFromBoundsGroup::CreateWidget()
{
 // ***
  cout << "vtkFiniteElementMRMLCreateBBFromBoundsGroup CreateWidget begun" << endl;
  
  if(this->IsCreated())
  {
    vtkErrorMacro("class already created");
    return;
  }

  // *** why is this here?  is this common for all kwwidgets? 
  //this->Superclass::CreateWidget();
  
  // *** 
  this->ObjectListComboBox = NULL;
  
  if(!this->ObjectListComboBox)  
  {
     this->ObjectListComboBox = vtkKWComboBoxWithLabel::New();
  }
  this->MainFrame->SetParent(this->GetParent());
  this->MainFrame->Create();
  this->MainFrame->SetLabelText("Create BB From Bounds");

  this->GetApplication()->Script(
    "pack %s -side top -anchor nw -expand y -padx 2 -pady 6", 
    this->MainFrame->GetWidgetName());

  ObjectListComboBox->SetParent(this->MainFrame->GetFrame());
  ObjectListComboBox->Create();
  ObjectListComboBox->SetLabelText("Object Selection : ");
  ObjectListComboBox->GetWidget()->ReadOnlyOn();

//   Slicer - the lines below caused a crash because the list was empty or NULL;  need to reference the MRML tree 
//   either directly or via an alternate implementation of the lists.  
    int i;
    for (i = 0; i < this->SurfaceList->GetNumberOfItems(); i++)
    {
      ObjectListComboBox->GetWidget()->AddValue(
        this->SurfaceList->GetItem(i)->GetFileName());
    }

  this->GetApplication()->Script(
    "pack %s -side top -anchor nw -expand y -padx 2 -pady 6", 
    ObjectListComboBox->GetWidgetName());

  this->DoneButton->SetParent(this->MainFrame->GetFrame());
  this->DoneButton->Create();
  this->DoneButton->SetText("Done");
  this->DoneButton->SetCommand(this, "CreateBBFromBoundsCallback");
  this->GetApplication()->Script(
    "pack %s -side left -anchor nw -expand y -padx 20 -pady 6", 
    this->DoneButton->GetWidgetName());

  this->CancelButton->SetParent(this->MainFrame->GetFrame());
  this->CancelButton->Create();
  this->CancelButton->SetText("Cancel");
  this->CancelButton->SetCommand(this, "CreateBBFromBoundsCancelCallback");
  this->GetApplication()->Script(
    "pack %s -side left -anchor nw -expand y -padx 2 -pady 6", 
    this->CancelButton->GetWidgetName());

}
//----------------------------------------------------------------------------
void vtkFiniteElementMRMLCreateBBFromBoundsGroup::Update()
{
  this->UpdateEnableState();
}
//---------------------------------------------------------------------------
void vtkFiniteElementMRMLCreateBBFromBoundsGroup::UpdateEnableState()
{
  this->Superclass::UpdateEnableState();
}

//----------------------------------------------------------------------------
void vtkFiniteElementMRMLCreateBBFromBoundsGroup::CreateBBFromBoundsCallback()
{
 cout << "vtkFiniteElementMRMLCreateBBFromBoundsGroup::CreateBBFromBoundsCallback" << endl;
  if(strcmp(this->ObjectListComboBox->GetWidget()->GetValue(),""))
  {
   
   cout << "got here: vtkFiniteElementMRMLCreateBBFromBoundsGroup::CreateBBFromBoundsCallback" << endl;
   
   // find the surface object named in the combo box and have its data referenced by the bbox 
   // instance we are creating.  This way the new bbox will correspond in shape to the selected surface
   // object.  The bbox instance will be used later as a source of data for the MRML node
   
    vtkKWComboBox *combobox = this->ObjectListComboBox->GetWidget();
    const char *name = combobox->GetValue();
    vtkPolyData *pdata = vtkMimxSurfacePolyDataActor::SafeDownCast(
        this->SurfaceList->GetItem(combobox->GetValueIndex(name)))->GetDataSet();
    vtkMimxBoundingBoxSource *bbox = vtkMimxBoundingBoxSource::New();
    bbox->SetSource(pdata);
    bbox->Update();
    cout << "MRMLCreateBB: copied bbox from surface" << endl;
    if (bbox->GetOutput())
    {

     //    vtkMimxUnstructuredGridActor::SafeDownCast(this->BBoxList->GetItem(
     //        this->BBoxList->GetNumberOfItems()-1))->SetObjectName("BBFromBounds_",Count);
     //    vtkMimxUnstructuredGridActor::SafeDownCast(this->BBoxList->GetItem(
     //        this->BBoxList->GetNumberOfItems()-1))->GetDataSet()->Modified();
     //    this->GetMimxViewWindow()->GetRenderWidget()->AddViewProp(
     //        this->BBoxList->GetItem(this->BBoxList->GetNumberOfItems()-1)->GetActor());
     //    this->GetMimxViewWindow()->GetRenderWidget()->Render();
     //    this->GetMimxViewWindow()->GetRenderWidget()->ResetCamera();
     //    this->ViewProperties->AddObjectList();

      // make an actor for the record, then pass to the MRML list
      vtkMimxUnstructuredGridActor *actor = vtkMimxUnstructuredGridActor::New();
      vtkFiniteElementBoundingBoxList *bblist = (vtkFiniteElementBoundingBoxList*)(this->BBoxList);
      
      // copy the ugrid from the boundingbox actor into the MRML node, so we can create bbox instances from it 
      // on demand.  Strobe the dataset so that a render of all observers will be forced
      vtkMimxUnstructuredGridActor::SafeDownCast(actor)->GetDataSet()->DeepCopy(bbox->GetOutput());
      vtkMimxUnstructuredGridActor::SafeDownCast(actor)->GetDataSet()->Modified();
             
      int sceneCount = bblist->GetNumberOfItems()+1;
      actor->SetObjectName("MRMLBBFromBounds_",sceneCount);

      // add the MRML node to the scene
      // *** the intentional subclass cast (defined a few lines above for bblist) is required. Why? 
      //***  since this method is derived through tcl, the regular polymorphism doesn't work, apparently. 
      //this->BBoxList->AppendItem(actor);
      bblist->AppendItem(actor);
   
        
      this->CreateBBFromBoundsCancelCallback();
      
      // add the bbox actor to the view window
      this->GetMimxViewWindow()->GetRenderWidget()->AddViewProp(actor->GetActor());
      this->GetMimxViewWindow()->GetRenderWidget()->Render();
      // add the name of the object to the properties browser
      this->ViewProperties->AddObjectList();
  
    }
    bbox->Delete();
  }
}
//----------------------------------------------------------------------------
void vtkFiniteElementMRMLCreateBBFromBoundsGroup::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
//----------------------------------------------------------------------------
void vtkFiniteElementMRMLCreateBBFromBoundsGroup::CreateBBFromBoundsCancelCallback()
{
//  this->MainFrame->UnpackChildren();
  this->GetApplication()->Script("pack forget %s", this->MainFrame->GetWidgetName());
  this->MenuGroup->SetMenuButtonsEnabled(1);
}
//------------------------------------------------------------------------------
void vtkFiniteElementMRMLCreateBBFromBoundsGroup::UpdateObjectLists()
{
  this->ObjectListComboBox->GetWidget()->DeleteAllValues();
  int i;
  for (i = 0; i < this->SurfaceList->GetNumberOfItems(); i++)
  {
    ObjectListComboBox->GetWidget()->AddValue(
      this->SurfaceList->GetItem(i)->GetFileName());
  }

}
//------------------------------------------------------------------------------
