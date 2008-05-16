/*=========================================================================

Program:   MIMX Meshing Toolkit
Module:    $RCSfile: vtkKWMimxViewProperties.cxx,v $
Language:  C++
Date:      $Date: 2008/04/16 23:29:58 $
Version:   $Revision: 1.31 $

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

#include "vtkKWMimxViewProperties.h"

#include "vtkLinkedListWrapper.h"
#include "vtkLinkedListWrapperTree.h"

#include "vtkKWApplication.h"
#include "vtkKWFrameWithLabel.h"
#include "vtkKWFrame.h"
#include "vtkKWIcon.h"
#include "vtkKWLabel.h"
#include "vtkKWMenu.h"
#include "vtkKWPushButton.h"
#include "vtkKWMultiColumnList.h"
#include "vtkKWMultiColumnListWithScrollbars.h"
#include "vtkRenderer.h"
#include "vtkKWRenderWidget.h"
#include "vtkKWTkUtilities.h"
#include "vtkCellData.h"
#include "vtkPointData.h"
#include "vtkUnstructuredGrid.h"
#include "vtkMimxUnstructuredGridActor.h"
#include "vtkMimxMeshActor.h"
#include "vtkKWFrameWithScrollbar.h"
#include "vtkKWCheckButton.h"
#include "vtkKWMimxViewPropertiesGroup.h"
#include "vtkKWDialog.h"


#include "vtkActor.h"
#include "vtkMath.h"
#include "vtkObjectFactory.h"
#include "vtkProperty.h"

#include <vtksys/stl/list>
#include <vtksys/stl/algorithm>
#include <vtksys/SystemTools.hxx>

// define the option types
#define VTK_KW_OPTION_NONE         0
#define VTK_KW_OPTION_LOAD                 1

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkKWMimxViewProperties);
vtkCxxRevisionMacro(vtkKWMimxViewProperties, "$Revision: 1.31 $");

//----------------------------------------------------------------------------
vtkKWMimxViewProperties::vtkKWMimxViewProperties()
{
  this->MultiColumnList = NULL;
  this->ObjectList = vtkLinkedListWrapper::New();
  this->MimxMainWindow = NULL;
  this->MainFrame = NULL;
  this->DoUndoTree = NULL;
  this->ViewPropertiesGroup = NULL;
  this->ViewButton = NULL;
  this->DisplayButton = NULL;
  this->ViewPropertyDialog = NULL;
  this->DisplayPropertyDialog = NULL;
}

//----------------------------------------------------------------------------
vtkKWMimxViewProperties::~vtkKWMimxViewProperties()
{
  if(this->MainFrame)
          this->MainFrame->Delete();
  if(this->ObjectList)
          this->ObjectList->Delete();
        if (this->ViewButton)
          this->ViewButton->Delete();
        if (this->DisplayButton)
          this->DisplayButton->Delete();
        if (this->ViewPropertyDialog)
          this->ViewPropertyDialog->Delete();
        if (this->DisplayPropertyDialog)
          this->DisplayPropertyDialog->Delete();
}
//----------------------------------------------------------------------------
void vtkKWMimxViewProperties::CreateWidget()
{
        if(this->IsCreated())
        {
                vtkErrorMacro("class already created");
                return;
        }
        this->Superclass::CreateWidget();
        if(!this->MainFrame)
        {
                this->MainFrame = vtkKWFrameWithLabel::New();
                //this->MainFrame = vtkKWFrameWithScrollbar::New();
        }
        this->MainFrame->SetParent(this->GetParent());
        this->MainFrame->Create();
        this->MainFrame->GetFrame()->SetReliefToGroove();
        this->MainFrame->SetLabelText("Objects");
  this->MainFrame->AllowFrameToCollapseOn();
        this->MainFrame->SetHeight(20);
  this->GetApplication()->Script(
    "pack %s -side top -anchor nw -expand y -padx 2 -pady 6 -fill x", 
    this->MainFrame->GetWidgetName());
  if(!this->MultiColumnList)
  {
    this->MultiColumnList = vtkKWMultiColumnListWithScrollbars::New();
  }
  this->MultiColumnList->SetParent(this->MainFrame->GetFrame());
  this->MultiColumnList->Create();
  this->MultiColumnList->GetWidget()->ClearStripeBackgroundColor();
  this->MultiColumnList->SetHorizontalScrollbarVisibility(0);
  this->MultiColumnList->GetWidget()->SetSelectionBackgroundColor(1.0,1.0,1.0);
  //this->MultiColumnList->SetHorizontalScrollbarVisibility(1);
  //this->MultiColumnList->SetVerticalScrollbarVisibility(1);
  //this->MultiColumnList->GetWidget()->SetHeight(5);
 // this->MultiColumnList->SetWidth(10);
//  this->MultiColumnList->GetWidget()->GetMovableColumnsOn();
//  this->MultiColumnList->SetPotentialCellColorsChangedCommand(
//    this->MultiColumnList, "ScheduleRefreshColorsOfAllCellsWithWindowCommand");
  int col_index;

  // Add the columns 
  
  col_index = this->MultiColumnList->GetWidget()->AddColumn(NULL);
  this->MultiColumnList->GetWidget()->SetColumnFormatCommandToEmptyOutput(col_index);
  this->MultiColumnList->GetWidget()->SetColumnLabelImageToPredefinedIcon(
          col_index, vtkKWIcon::IconEye);
        this->MultiColumnList->GetWidget()->SetColumnWidth( col_index, 2);
        
        col_index = this->MultiColumnList->GetWidget()->AddColumn("Name");
  this->MultiColumnList->GetWidget()->SetColumnWidth( col_index, 30);
  this->MultiColumnList->GetWidget()->SetColumnFormatCommandToEmptyOutput(col_index);
  
           
  
 //this->MultiColumnList->InsertCellText(0, 0, "");
 // this->MultiColumnList->InsertCellTextAsInt(0, 1, 0);
 // this->MultiColumnList->SetCellWindowCommandToCheckButton(0, 1);
 // this->MultiColumnList->GetCellWindowAsCheckButton(0,1)->SetEnabled(0);
  this->MultiColumnList->GetWidget()->SetSortArrowVisibility(0);
  this->MultiColumnList->GetWidget()->ColumnSeparatorsVisibilityOff();
  this->MultiColumnList->GetWidget()->SetHeight( 4 );
  
//  this->MultiColumnList->SetSelectioModeToSingle();
//  this->MultiColumnList->RowSeparatorsVisibilityOff();
  this->GetApplication()->Script(
    "pack %s -side top -anchor nw -expand n -padx 2 -pady 2 -fill x", 
    this->MultiColumnList->GetWidgetName());

/*  
  if(!this->ViewButton)
  {
    this->ViewButton = vtkKWPushButton::New();
  }
  this->ViewButton->SetParent(this->MainFrame->GetFrame());
  this->ViewButton->Create();
  this->ViewButton->SetText("View");
  this->ViewButton->SetCommand(this, "DisplayPropertyCallback");
  this->GetApplication()->Script(
          "pack %s -side left -anchor nw -expand y -padx 20 -pady 6", 
          this->ViewButton->GetWidgetName());

  if(!this->DisplayButton)
  {
    this->DisplayButton = vtkKWPushButton::New();
  }
  this->DisplayButton->SetParent(this->MainFrame->GetFrame());
  this->DisplayButton->Create();
  this->DisplayButton->SetText("Display");
  this->DisplayButton->SetCommand(this, "ViewPropertyCallback");
  this->GetApplication()->Script(
    "pack %s -side right -anchor ne -expand y -padx 20 -pady 6", 
    this->DisplayButton->GetWidgetName());
*/    
}
//----------------------------------------------------------------------------
void vtkKWMimxViewProperties::Update()
{
        this->UpdateEnableState();
}
//---------------------------------------------------------------------------
void vtkKWMimxViewProperties::UpdateEnableState()
{
        this->Superclass::UpdateEnableState();
}
//----------------------------------------------------------------------------
void vtkKWMimxViewProperties::VisibilityCallback(int flag)
{
  for (int i=0; i<this->ObjectList->GetNumberOfItems(); i++)
  {
    if(this->MultiColumnList->GetWidget()->GetCellWindowAsCheckButton(i,0)->GetSelectedState())
    {
      if(this->ObjectList->GetItem(i)->GetDataType() == ACTOR_FE_MESH)
      {
        vtkMimxMeshActor *meshActor = vtkMimxMeshActor::SafeDownCast(this->ObjectList->GetItem(i));
        meshActor->ShowMesh();
      }
      else
      {
        this->GetMimxMainWindow()->GetRenderWidget()->AddViewProp(
          this->ObjectList->GetItem(i)->GetActor());
      }
    }
        else
        {
                  if(this->ObjectList->GetItem(i)->GetDataType() == ACTOR_FE_MESH)
      {
        vtkMimxMeshActor *meshActor = vtkMimxMeshActor::SafeDownCast(this->ObjectList->GetItem(i));
        meshActor->HideMesh();
      }
      else
      {
        this->GetMimxMainWindow()->GetRenderWidget()->RemoveViewProp(
                          this->ObjectList->GetItem(i)->GetActor());
                        }
        }
  }
  this->GetMimxMainWindow()->GetRenderWidget()->Render();
  this->GetMimxMainWindow()->GetRenderWidget()->ResetCamera();
}
//----------------------------------------------------------------------------
void vtkKWMimxViewProperties::AddObjectList(vtkMimxActorBase *actor)
{

        this->ObjectList->AppendItem(actor);
        
        /* Add the Object to the Display MultiColumnList */
        int rowIndex = this->ObjectList->GetNumberOfItems()-1;
        
        this->MultiColumnList->GetWidget()->InsertCellTextAsInt(rowIndex, 0, 1);
  this->MultiColumnList->GetWidget()->SetCellWindowCommandToCheckButton( rowIndex, 0);
        this->MultiColumnList->GetWidget()->GetCellWindowAsCheckButton(rowIndex,0)->SetCommand(this, "VisibilityCallback");
        
  this->MultiColumnList->GetWidget()->InsertCellText(rowIndex, 1, this->ObjectList->GetItem(rowIndex)->GetFileName());
  
  this->MultiColumnList->GetWidget()->SetCellWindowCommand(rowIndex, 1, this, "CreateNameCellCallback");
  
  //this->MultiColumnList->GetWidget()->InsertCellText(rowIndex, 1, "");
  
  //InsertCellText(rowIndex, 1, this->ObjectList->GetItem(rowIndex)->GetFileName());

        //this->MultiColumnList->GetWidget()->SetSelectionChangedCommand(NULL, "SetViewProperties");
  
        this->GetMimxMainWindow()->GetRenderWidget()->AddViewProp(actor->GetActor());
        this->GetMimxMainWindow()->GetRenderWidget()->Render();
        this->GetMimxMainWindow()->GetRenderWidget()->ResetCamera();

//      this->UpdateVisibility();
}
//----------------------------------------------------------------------------
void vtkKWMimxViewProperties::UpdateVisibility()
{
        for (int i=0; i<this->ObjectList->GetNumberOfItems(); i++)
        {
                if(this->GetMimxMainWindow()->GetRenderWidget()->GetRenderer()->HasViewProp(
                        this->ObjectList->GetItem(i)->GetActor()))
                        this->MultiColumnList->GetWidget()->GetCellWindowAsCheckButton(i,0)->
                        SetSelectedState(1);
                else
                        this->MultiColumnList->GetWidget()->GetCellWindowAsCheckButton(i,0)->
                        SetSelectedState(0);
        }
}
//----------------------------------------------------------------------------
void vtkKWMimxViewProperties::DeleteObjectList(int DataType, int Position)
{
        // match the position from one list to the other
        int i, currpos;
        int poscount = 0;

        for(i=0; i< this->ObjectList->GetNumberOfItems(); i++)
        {
                if(this->ObjectList->GetItem(i)->GetDataType() == DataType)
                {
                        if(poscount == Position)
                        {
                                currpos = i;
                                break;
                        }
                        poscount ++;
                }
        }
        this->MultiColumnList->GetWidget()->DeleteRow(currpos);
        if(DataType == 6)
        {
                vtkMimxMeshActor::SafeDownCast(
                        this->ObjectList->GetItem(currpos))->HideMesh();
        }
        else
        {
                this->GetMimxMainWindow()->GetRenderWidget()->RemoveViewProp(
                        this->ObjectList->GetItem(currpos)->GetActor());
        }
        this->ObjectList->RemoveItem(currpos);
        this->GetMimxMainWindow()->GetRenderWidget()->Render();
        this->GetMimxMainWindow()->GetRenderWidget()->ResetCamera();
}
//----------------------------------------------------------------------------
void vtkKWMimxViewProperties::DeleteObjectList(const char *name)
{
        // match the position from one list to the other
        int i;  
        for (i=0; i<this->ObjectList->GetNumberOfItems(); i++)
        {
                const char *name1 = this->MultiColumnList->GetWidget()->GetCellText(i,1);
                if(!strcmp(name, this->MultiColumnList->GetWidget()->GetCellText(i,1)))
                {
                        break;
                }
        }
        this->MultiColumnList->GetWidget()->DeleteRow(i);
        this->GetMimxMainWindow()->GetRenderWidget()->RemoveViewProp(
                this->ObjectList->GetItem(i)->GetActor());
        this->ObjectList->RemoveItem(i);
        this->GetMimxMainWindow()->GetRenderWidget()->Render();
        this->GetMimxMainWindow()->GetRenderWidget()->ResetCamera();
}
//----------------------------------------------------------------------------
void vtkKWMimxViewProperties::ViewPropertyCallback(int row)
{
        if (!this->ViewPropertyDialog)
        {
          this->ViewPropertyDialog = vtkKWMimxViewPropertiesGroup::New();
        }
          this->ViewPropertyDialog->SetMimxMainWindow(this->GetMimxMainWindow());
          this->ViewPropertyDialog->SetMultiColumnList(this->MultiColumnList);
          this->ViewPropertyDialog->SetObjectList(
                  this->GetMimxMainWindow()->GetViewProperties()->GetObjectList());
          this->ViewPropertyDialog->SetApplication(this->GetApplication());
          this->MultiColumnList->GetWidget()->SetSelectionChangedCommand(this->ViewPropertyDialog, "SetViewProperties");
          this->ViewPropertyDialog->SetSelectionRow(row);
          this->ViewPropertyDialog->Create();
          this->ViewPropertyDialog->SetViewProperties();  
          this->ViewPropertyDialog->Display();
}
//----------------------------------------------------------------------------
void vtkKWMimxViewProperties::DisplayPropertyCallback()
{
        if (!this->DisplayPropertyDialog)
        {
          this->DisplayPropertyDialog = vtkKWMimxDisplayPropertiesGroup::New();
          this->DisplayPropertyDialog->SetMimxMainWindow(this->GetMimxMainWindow());
          this->DisplayPropertyDialog->SetApplication(this->GetApplication());
          this->DisplayPropertyDialog->Create();
        }
        this->DisplayPropertyDialog->Display();
        
}
//----------------------------------------------------------------------------
void vtkKWMimxViewProperties::CreateNameCellCallback(const char *tableWidgetName, int row, int col, const char *widgetName)
{
  vtkKWPushButton *cellButton = vtkKWPushButton::New();
  cellButton->SetWidgetName(widgetName);
  cellButton->SetApplication(this->GetApplication());
  cellButton->Create();
  cellButton->SetFont("arial 8 bold");
  cellButton->SetReliefToFlat();
  cellButton->SetBorderWidth(0);
  cellButton->SetText(this->MultiColumnList->GetWidget()->GetCellText(row, col));
  this->MultiColumnList->GetWidget()->InsertCellText(row, col, "");
  char calbackString[256];
  sprintf(calbackString, "ViewPropertyCallback %d", row);
//  cellButton->SetCommand(this, "ViewPropertyCallback");
  cellButton->SetCommand(this, calbackString);
  
//  this->MultiColumnList->GetWidget()->RefreshColorsOfCellWithWindowCommand(row, col);
//  this->MultiColumnList->GetWidget()->RefreshAllCellsWithWindowCommand ();
//  this->MultiColumnList->GetWidget()->ScheduleRefreshColorsOfAllCellsWithWindowCommand();
  /*
  app->Script(
    "proc CreateCompletionCellCallback {tw row col w} { "
    "  frame $w -bg #882233 -relief groove -bd 2 -height 10 -width [expr [%s GetCellTextAsDouble $row $col] * 0.01 * 70] ;"
    "  %s AddBindingsToWidgetName $w "
    "}", mcl1->GetTclName(), mcl1->GetTclName());
  */
}    
//----------------------------------------------------------------------------
void vtkKWMimxViewProperties::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
//----------------------------------------------------------------------------
