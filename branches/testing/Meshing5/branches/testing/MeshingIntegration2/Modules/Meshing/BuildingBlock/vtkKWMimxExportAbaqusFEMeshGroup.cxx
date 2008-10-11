/*=========================================================================

Program:   MIMX Meshing Toolkit
Module:    $RCSfile: vtkKWMimxExportAbaqusFEMeshGroup.cxx,v $
Language:  C++
Date:      $Date: 2008/04/25 21:31:09 $
Version:   $Revision: 1.20 $

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

#include "vtkKWMimxExportAbaqusFEMeshGroup.h"
#include "vtkKWMimxMainWindow.h"
#include "vtkKWMimxMainMenuGroup.h"
#include "vtkMimxErrorCallback.h"
#include "vtkKWMimxMainNotebook.h"

#include "vtkLinkedListWrapper.h"

#include "vtkActor.h"
#include "vtkMimxUnstructuredGridActor.h"
#include "vtkUnstructuredGrid.h"
#include "vtkRenderer.h"
#include "vtkObjectFactory.h"
#include "vtkKWRenderWidget.h"


#include "vtkKWApplication.h"
#include "vtkKWLoadSaveDialog.h"
#include "vtkKWEvent.h"
#include "vtkKWFrame.h"
#include "vtkKWFrameWithLabel.h"
#include "vtkKWTextWithScrollbarsWithLabel.h"
#include "vtkKWComboBoxWithLabel.h"
#include "vtkKWComboBox.h"
#include "vtkKWPushButton.h"
#include "vtkKWMimxMainUserInterfacePanel.h"
#include "vtkMimxMeshActor.h"
#include "vtkKWTextWithScrollbars.h"
#include "vtkKWText.h"

#include "vtkMimxAbaqusFileWriter.h"

#include <vtksys/stl/list>
#include <vtksys/stl/algorithm>
#include <vtksys/SystemTools.hxx>

// define the option types
#define VTK_KW_OPTION_NONE         0
#define VTK_KW_OPTION_LOAD                 1

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkKWMimxExportAbaqusFEMeshGroup);
vtkCxxRevisionMacro(vtkKWMimxExportAbaqusFEMeshGroup, "$Revision: 1.20 $");

//----------------------------------------------------------------------------
vtkKWMimxExportAbaqusFEMeshGroup::vtkKWMimxExportAbaqusFEMeshGroup()
{
  this->ObjectListComboBox = NULL;
  this->FileBrowserDialog = NULL;
  this->HeaderInformationText = NULL;
}

//----------------------------------------------------------------------------
vtkKWMimxExportAbaqusFEMeshGroup::~vtkKWMimxExportAbaqusFEMeshGroup()
{
  if(this->ObjectListComboBox)
     this->ObjectListComboBox->Delete();
  if(this->FileBrowserDialog)
          this->FileBrowserDialog->Delete();
  if(this->HeaderInformationText)
          this->HeaderInformationText->Delete();
 }
//----------------------------------------------------------------------------
void vtkKWMimxExportAbaqusFEMeshGroup::CreateWidget()
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
  this->MainFrame->SetLabelText("Save FE Mesh (Abaqus file format)");

  this->GetApplication()->Script(
    "pack %s -side top -anchor nw -expand n -padx 2 -pady 0 -fill x", 
    this->MainFrame->GetWidgetName());

  ObjectListComboBox->SetParent(this->MainFrame->GetFrame());
  ObjectListComboBox->Create();
  ObjectListComboBox->SetLabelText("Mesh : ");
  ObjectListComboBox->GetWidget()->ReadOnlyOn();
  this->GetApplication()->Script(
    "pack %s -side top -anchor nw -expand y -padx 2 -pady 6 -fill x", 
    ObjectListComboBox->GetWidgetName());

  this->HeaderInformationText = vtkKWTextWithScrollbarsWithLabel::New();
  this->HeaderInformationText->SetParent(this->MainFrame->GetFrame());
  this->HeaderInformationText->Create();
  this->HeaderInformationText->SetLabelPositionToTop( );
  this->HeaderInformationText->SetLabelText("Header Information");
  this->GetApplication()->Script(
          "pack %s -side top -anchor nw -expand y -padx 2 -pady 6 -fill x", 
          this->HeaderInformationText->GetWidgetName());
    
  this->ApplyButton->SetParent(this->MainFrame->GetFrame());
  this->ApplyButton->Create();
  this->ApplyButton->SetText("Apply");
  this->ApplyButton->SetCommand(this, "ExportAbaqusFEMeshApplyCallback");
  this->GetApplication()->Script(
          "pack %s -side left -anchor nw -expand y -padx 20 -pady 6", 
          this->ApplyButton->GetWidgetName());

  this->CancelButton->SetParent(this->MainFrame->GetFrame());
  this->CancelButton->Create();
  this->CancelButton->SetText("Cancel");
  this->CancelButton->SetCommand(this, "ExportAbaqusFEMeshCancelCallback");
  this->GetApplication()->Script(
    "pack %s -side right -anchor ne -expand y -padx 20 -pady 6", 
    this->CancelButton->GetWidgetName());

}
//----------------------------------------------------------------------------
void vtkKWMimxExportAbaqusFEMeshGroup::Update()
{
        this->UpdateEnableState();
}
//---------------------------------------------------------------------------
void vtkKWMimxExportAbaqusFEMeshGroup::UpdateEnableState()
{
        this->UpdateObjectLists();
        this->Superclass::UpdateEnableState();
}
//----------------------------------------------------------------------------
int vtkKWMimxExportAbaqusFEMeshGroup::ExportAbaqusFEMeshApplyCallback()
{
        vtkMimxErrorCallback *callback = this->GetMimxMainWindow()->GetErrorCallback();
        callback->SetState(0);
  if(!strcmp(this->ObjectListComboBox->GetWidget()->GetValue(),""))
  {
          callback->ErrorMessage("FE mesh selection required");
          return 0;
  }
  vtkKWComboBox *combobox = this->ObjectListComboBox->GetWidget();
    const char *name = combobox->GetValue();

        int num = combobox->GetValueIndex(name);
        if(num < 0 || num > combobox->GetNumberOfValues()-1)
        {
                callback->ErrorMessage("Choose valid FE Mesh");
                combobox->SetValue("");
                return 0;
        }

  vtkUnstructuredGrid *ugrid = vtkMimxMeshActor::SafeDownCast(
          this->FEMeshList->GetItem(combobox->GetValueIndex(name)))->GetDataSet();
        if(!this->FileBrowserDialog)
        {
                this->FileBrowserDialog = vtkKWLoadSaveDialog::New() ;
                this->FileBrowserDialog->SaveDialogOn();
                this->FileBrowserDialog->SetApplication(this->GetApplication());
//              dialog->SetParent(this->RenderWidget->GetParentTopLevel()) ;
                this->FileBrowserDialog->Create();
                this->FileBrowserDialog->RetrieveLastPathFromRegistry("LastPath");
                this->FileBrowserDialog->SetTitle ("Save FE Mesh (Abaqus File format)");
                this->FileBrowserDialog->SetFileTypes ("{{INP files} {.inp}}");
                this->FileBrowserDialog->SetDefaultExtension (".inp");
        }
        this->FileBrowserDialog->Invoke();
        if(this->FileBrowserDialog->GetStatus() == vtkKWDialog::StatusOK)
        {
                if(this->FileBrowserDialog->GetFileName())
                {
                        const char *filename = FileBrowserDialog->GetFileName();
                        this->GetApplication()->SetRegistryValue(
                                1, "RunTime", "LastPath", vtksys::SystemTools::GetFilenamePath( filename ).c_str());
                        this->FileBrowserDialog->SaveLastPathToRegistry("LastPath");

                        vtkMimxAbaqusFileWriter *writer = vtkMimxAbaqusFileWriter::New();
                        writer->SetHeaderInformation( 
                                this->HeaderInformationText->GetWidget()->GetWidget()->GetText() );
                        writer->SetFileName( filename );
                        writer->SetInput(ugrid);
                        writer->AddObserver(vtkCommand::ErrorEvent, callback, 1.0);
                        writer->Update();
                        writer->RemoveObserver(callback);
                        writer->Delete();
                        
                        this->GetMimxMainWindow()->SetStatusText("Exported Abaqus Mesh");
                        
                        return 1;
                }
        }
        return 0;
}
//----------------------------------------------------------------------------
void vtkKWMimxExportAbaqusFEMeshGroup::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
//----------------------------------------------------------------------------
void vtkKWMimxExportAbaqusFEMeshGroup::ExportAbaqusFEMeshCancelCallback()
{
//  this->MainFrame->UnpackChildren();
  this->GetApplication()->Script("pack forget %s", this->MainFrame->GetWidgetName());
  this->MenuGroup->SetMenuButtonsEnabled(1);
    this->GetMimxMainWindow()->GetMainUserInterfacePanel()->GetMimxMainNotebook()->SetEnabled(1);
}
//------------------------------------------------------------------------------
void vtkKWMimxExportAbaqusFEMeshGroup::UpdateObjectLists()
{
        this->ObjectListComboBox->GetWidget()->DeleteAllValues();
        
        int defaultItem = -1;
        for (int i = 0; i < this->FEMeshList->GetNumberOfItems(); i++)
        {
                ObjectListComboBox->GetWidget()->AddValue(
                        this->FEMeshList->GetItem(i)->GetFileName());
                        
                int viewedItem = this->GetMimxMainWindow()->GetRenderWidget()->GetRenderer()->HasViewProp(
                        this->FEMeshList->GetItem(i)->GetActor());
                if ( (defaultItem == -1) && ( viewedItem ) )
                {
                  defaultItem = i;
                }
        }
        
        if (defaultItem != -1)
  {
    ObjectListComboBox->GetWidget()->SetValue(
          this->FEMeshList->GetItem(defaultItem)->GetFileName());
  }
  this->HeaderInformationText->GetWidget()->GetWidget()->SetText("");
}
//--------------------------------------------------------------------------------
void vtkKWMimxExportAbaqusFEMeshGroup::ExportAbaqusFEMeshDoneCallback()
{
        if(this->ExportAbaqusFEMeshApplyCallback())
                this->ExportAbaqusFEMeshCancelCallback();
}
//---------------------------------------------------------------------------------
