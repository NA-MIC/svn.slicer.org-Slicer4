/*=========================================================================

Program:   MIMX Meshing Toolkit
Module:    $RCSfile: vtkKWMimxGroupBase.cxx,v $
Language:  C++
Date:      $Date: 2008/02/23 16:16:23 $
Version:   $Revision: 1.13 $

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

#include "vtkKWMimxGroupBase.h"

#include "vtkKWMimxMainMenuGroup.h"

#include "vtkMimxUnstructuredGridWidget.h"

#include "vtkActor.h"
#include "vtkMimxBoundingBoxSource.h"
#include "vtkMimxSurfacePolyDataActor.h"
#include "vtkPolyData.h"
#include "vtkMimxUnstructuredGridActor.h"
#include "vtkUnstructuredGrid.h"
#include "vtkKWRenderWidget.h"
//#include "vtkKWMimxViewProperties.h"

#include "vtkKWApplication.h"
#include "vtkKWFileBrowserDialog.h"
#include "vtkKWEvent.h"
#include "vtkKWFrame.h"
#include "vtkKWFrameWithLabel.h"
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
#include "vtkLinkedListWrapper.h"
#include "vtkMath.h"
#include "vtkObjectFactory.h"
#include "vtkKWPushButton.h"
#include "vtkKWComboBoxWithLabel.h"
#include "vtkKWComboBox.h"
#include "vtkLinkedListWrapperTree.h"

#include <vtksys/stl/list>
#include <vtksys/stl/algorithm>

// define the option types
#define VTK_KW_OPTION_NONE         0
#define VTK_KW_OPTION_LOAD                 1

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkKWMimxGroupBase);
vtkCxxRevisionMacro(vtkKWMimxGroupBase, "$Revision: 1.13 $");

//----------------------------------------------------------------------------
vtkKWMimxGroupBase::vtkKWMimxGroupBase()
{
  this->BBoxList = NULL;
  this->FEMeshList = NULL;
  this->SurfaceList = NULL;
  this->ImageList = NULL;
  this->MimxMainWindow = NULL;
  this->CancelButton = vtkKWPushButton::New();
  this->DoneButton = vtkKWPushButton::New();
  this->MainFrame = vtkKWFrameWithLabel::New();
  this->Count = 0;
  this->ViewProperties = NULL;
  this->MenuGroup = NULL;
  this->ApplyButton = vtkKWPushButton::New();
  this->DoUndoTree = NULL;
}

//----------------------------------------------------------------------------
vtkKWMimxGroupBase::~vtkKWMimxGroupBase()
{
  this->CancelButton->Delete();
  this->DoneButton->Delete();
  this->MainFrame->Delete();
  this->ApplyButton->Delete();
}
//----------------------------------------------------------------------------
void vtkKWMimxGroupBase::CreateWidget()
{
        if(this->IsCreated())
        {
                vtkErrorMacro("class already created");
                return;
        }

        this->Superclass::CreateWidget();

}
//----------------------------------------------------------------------------
void vtkKWMimxGroupBase::Update()
{
        this->UpdateEnableState();
}
//---------------------------------------------------------------------------
void vtkKWMimxGroupBase::UpdateEnableState()
{
        this->Superclass::UpdateEnableState();
}
//----------------------------------------------------------------------------
void vtkKWMimxGroupBase::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
//----------------------------------------------------------------------------
