/*=========================================================================

Program:   MIMX Meshing Toolkit
Module:    $RCSfile: vtkKWMimxSaveVTKBBGroup.cxx,v $
Language:  C++
Date:      $Date: 2008/04/25 21:31:09 $
Version:   $Revision: 1.17 $

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

#include "vtkKWMimxSaveVTKBBGroup.h"
#include "vtkKWMimxMainWindow.h"
#include "vtkKWMimxMainMenuGroup.h"
#include "vtkMimxErrorCallback.h"
#include "vtkKWMimxMainNotebook.h"

#include "vtkLinkedListWrapper.h"

#include "vtkActor.h"
#include "vtkMimxUnstructuredGridActor.h"
#include "vtkUnstructuredGrid.h"

#include "vtkKWApplication.h"
#include "vtkKWLoadSaveDialog.h"
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


#include "vtkUnstructuredGridWriter.h"

#include <vtksys/stl/list>
#include <vtksys/stl/algorithm>
#include <vtksys/SystemTools.hxx>

// define the option types
#define VTK_KW_OPTION_NONE         0
#define VTK_KW_OPTION_LOAD                 1

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkKWMimxSaveVTKBBGroup);
vtkCxxRevisionMacro(vtkKWMimxSaveVTKBBGroup, "$Revision: 1.17 $");

//----------------------------------------------------------------------------
vtkKWMimxSaveVTKBBGroup::vtkKWMimxSaveVTKBBGroup()
{
  this->ObjectListComboBox = NULL;
  this->FileBrowserDialog = NULL;
}

//----------------------------------------------------------------------------
vtkKWMimxSaveVTKBBGroup::~vtkKWMimxSaveVTKBBGroup()
{
  if(this->ObjectListComboBox)
     this->ObjectListComboBox->Delete();
  if(this->FileBrowserDialog)
          this->FileBrowserDialog->Delete();
 }
//----------------------------------------------------------------------------
void vtkKWMimxSaveVTKBBGroup::CreateWidget()
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
  this->MainFrame->SetLabelText("Save BB (VTK file format)");

  this->GetApplication()->Script(
    "pack %s -side top -anchor nw -expand n -padx 2 -pady 6 -fill x", 
    this->MainFrame->GetWidgetName());

  ObjectListComboBox->SetParent(this->MainFrame->GetFrame());
  ObjectListComboBox->Create();
  ObjectListComboBox->SetLabelText("Building Block : ");
  ObjectListComboBox->GetWidget()->ReadOnlyOn();
  this->GetApplication()->Script(
    "pack %s -side top -anchor nw -expand y -padx 2 -pady 6 -fill x", 
    ObjectListComboBox->GetWidgetName());

  this->ApplyButton->SetParent(this->MainFrame->GetFrame());
  this->ApplyButton->Create();
  this->ApplyButton->SetText("Apply");
  this->ApplyButton->SetCommand(this, "SaveVTKBBApplyCallback");
  this->GetApplication()->Script(
          "pack %s -side left -anchor nw -expand y -padx 20 -pady 6", 
          this->ApplyButton->GetWidgetName());
/*
  this->DoneButton->SetParent(this->MainFrame->GetFrame());
  this->DoneButton->Create();
  this->DoneButton->SetText("Done");
  this->DoneButton->SetCommand(this, "SaveVTKBBDoneCallback");
  this->GetApplication()->Script(
    "pack %s -side left -anchor nw -expand y -padx 20 -pady 6", 
    this->DoneButton->GetWidgetName());
*/
  this->CancelButton->SetParent(this->MainFrame->GetFrame());
  this->CancelButton->Create();
  this->CancelButton->SetText("Cancel");
  this->CancelButton->SetCommand(this, "SaveVTKBBCancelCallback");
  this->GetApplication()->Script(
    "pack %s -side right -anchor ne -expand y -padx 20 -pady 6", 
    this->CancelButton->GetWidgetName());

}
//----------------------------------------------------------------------------
void vtkKWMimxSaveVTKBBGroup::Update()
{
        this->UpdateEnableState();
}
//---------------------------------------------------------------------------
void vtkKWMimxSaveVTKBBGroup::UpdateEnableState()
{
        this->UpdateObjectLists();
        this->Superclass::UpdateEnableState();
}
//----------------------------------------------------------------------------
int vtkKWMimxSaveVTKBBGroup::SaveVTKBBApplyCallback()
{
        vtkMimxErrorCallback *callback = this->GetMimxMainWindow()->GetErrorCallback();
        if(!strcmp(this->ObjectListComboBox->GetWidget()->GetValue(),""))
        {
                callback->ErrorMessage("Building Block selection required");
                return 0;
        }

  vtkKWComboBox *combobox = this->ObjectListComboBox->GetWidget();
    const char *name = combobox->GetValue();

        int num = combobox->GetValueIndex(name);
        if(num < 0 || num > combobox->GetNumberOfValues()-1)
        {
                callback->ErrorMessage("Choose valid Building-block structure");
                combobox->SetValue("");
                return 0;
        }

  vtkUnstructuredGrid *ugrid = vtkMimxUnstructuredGridActor::SafeDownCast(
          this->BBoxList->GetItem(combobox->GetValueIndex(name)))->GetDataSet();
        if(!this->FileBrowserDialog)
        {
                this->FileBrowserDialog = vtkKWLoadSaveDialog::New() ;
                this->FileBrowserDialog->SaveDialogOn();
                this->FileBrowserDialog->SetApplication(this->GetApplication());
//              dialog->SetParent(this->RenderWidget->GetParentTopLevel()) ;
                this->FileBrowserDialog->Create();
                this->FileBrowserDialog->RetrieveLastPathFromRegistry("FEMeshDataPath");
                this->FileBrowserDialog->SetTitle ("Save BB (VTK Unstructured grid File format)");
                this->FileBrowserDialog->SetFileTypes ("{{VTK files} {.vtk}}");
                this->FileBrowserDialog->SetDefaultExtension (".vtk");
        }
        this->FileBrowserDialog->RetrieveLastPathFromRegistry("LastPath");
        this->FileBrowserDialog->Invoke();
        if(this->FileBrowserDialog->GetStatus() == vtkKWDialog::StatusOK)
        {
                if(this->FileBrowserDialog->GetFileName())
                {
                        const char *filename = FileBrowserDialog->GetFileName();
                        this->GetApplication()->SetRegistryValue(
                                1, "RunTime", "LastPath", vtksys::SystemTools::GetFilenamePath( filename ).c_str());
                        this->FileBrowserDialog->SaveLastPathToRegistry("LastPath");
                        vtkUnstructuredGridWriter *writer = vtkUnstructuredGridWriter::New();
                        writer->SetFileName(this->FileBrowserDialog->GetFileName());
                        writer->SetInput(ugrid);
                        writer->Update();
                        writer->Delete();
                        
                        this->GetMimxMainWindow()->SetStatusText("Saved Building Block");
                        
                        return 1;
                }
        }
        return 0;
}
//----------------------------------------------------------------------------
void vtkKWMimxSaveVTKBBGroup::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
//----------------------------------------------------------------------------
void vtkKWMimxSaveVTKBBGroup::SaveVTKBBCancelCallback()
{
//  this->MainFrame->UnpackChildren();
  this->GetApplication()->Script("pack forget %s", this->MainFrame->GetWidgetName());
  this->MenuGroup->SetMenuButtonsEnabled(1);
    this->GetMimxMainWindow()->GetMainUserInterfacePanel()->GetMimxMainNotebook()->SetEnabled(1);
}
//-----------------------------------------------------------------------------
void vtkKWMimxSaveVTKBBGroup::UpdateObjectLists()
{
        this->ObjectListComboBox->GetWidget()->DeleteAllValues();
        
        int defaultItem = -1;
        for (int i = 0; i < this->BBoxList->GetNumberOfItems(); i++)
        {
                ObjectListComboBox->GetWidget()->AddValue(
                        this->BBoxList->GetItem(i)->GetFileName());
          int viewedItem = this->GetMimxMainWindow()->GetRenderWidget()->GetRenderer()->HasViewProp(
                                         this->BBoxList->GetItem(i)->GetActor());
    if ( (defaultItem == -1) && ( viewedItem ) )
                {
                  defaultItem = i;
                }                               
        }
        if (defaultItem != -1)
  {
    ObjectListComboBox->GetWidget()->SetValue(
          this->BBoxList->GetItem(defaultItem)->GetFileName());
  }
}
//--------------------------------------------------------------------------------
void vtkKWMimxSaveVTKBBGroup::SaveVTKBBDoneCallback()
{
        if(this->SaveVTKBBApplyCallback())
                this->SaveVTKBBCancelCallback();
}
//---------------------------------------------------------------------------------
