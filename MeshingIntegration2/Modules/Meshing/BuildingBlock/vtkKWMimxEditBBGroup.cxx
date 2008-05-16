/*=========================================================================

Program:   MIMX Meshing Toolkit
Module:    $RCSfile: vtkKWMimxEditBBGroup.cxx,v $
Language:  C++
Date:      $Date: 2008/05/05 19:30:08 $
Version:   $Revision: 1.61 $

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

#include "vtkKWMimxEditBBGroup.h"
#include "vtkKWMimxMainWindow.h"
#include "vtkKWMimxMainNotebook.h"

#include "vtkMimxUnstructuredGridWidget.h"
#include "vtkMimxExtractEdgeWidget.h"
#include "vtkMimxExtractFaceWidget.h"
#include "vtkMimxExtractCellWidget.h"
#include "vtkMimxErrorCallback.h"
#include "vtkMimxMapOriginalCellAndPointIds.h"
#include "vtkMimxExtractSurface.h"
#include "vtkMimxExtrudePolyData.h"
#include "vtkDataSet.h"
#include "vtkAppendFilter.h"
#include "vtkKWCheckButtonWithLabel.h"
#include "vtkMimxMirrorUnstructuredHexahedronGridCell.h"

#include "vtkActor.h"
#include "vtkMimxBoundingBoxSource.h"
#include "vtkMimxSurfacePolyDataActor.h"
#include "vtkPolyData.h"
#include "vtkMimxUnstructuredGridActor.h"
#include "vtkUnstructuredGrid.h"
#include "vtkMimxSplitUnstructuredHexahedronGridCell.h"
#include "vtkMimxAddUnstructuredHexahedronGridCell.h"
#include "vtkMimxDeleteUnstructuredHexahedronGridCell.h"
#include "vtkMimxSelectCellsWidget.h"
#include "vtkIdList.h"
#include "vtkKWMimxMergeBBGroup.h"
#include "vtkKWMimxEditBBGroup.h"
#include "vtkMimxSelectCellsWidget.h"
#include "vtkMimxSubdivideBoundingBox.h"
#include "vtkKWPushButtonSet.h"
#include "vtkMergeCells.h"
#include "vtkCellData.h"

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
#include "vtkKWEntryWithLabel.h"
#include "vtkKWFrame.h"
#include "vtkLinkedListWrapper.h"
#include "vtkPlaneWidget.h"
#include "vtkMath.h"
#include "vtkPlane.h"
#include "vtkProperty.h"
#include "vtkRenderer.h"
#include "vtkKWCheckButtonSet.h"
#include "vtkKWCheckButton.h"

#include "vtkUnstructuredGridWriter.h"
#include "vtkPolyDataWriter.h"

#include "vtkObjectFactory.h"
#include "vtkKWPushButton.h"
#include "vtkKWComboBoxWithLabel.h"
#include "vtkKWComboBox.h"
#include "vtkKWEntryWithLabel.h"
#include "vtkKWRadioButton.h"
#include "vtkKWRadioButtonSet.h"
#include "vtkKWMimxMainUserInterfacePanel.h"


#include "Resources/mimxRedo.h"
#include "Resources/mimxUndo.h"
#include "Resources/mimxAdd.h"
#include "Resources/mimxDelete.h"
#include "Resources/mimxSplit.h"
#include "Resources/mimxMove.h"
#include "Resources/mimxMirror.h"
#include "Resources/mimxMerge.h"

#include <vtksys/stl/list>
#include <vtksys/stl/algorithm>

// define the option types
#define VTK_KW_OPTION_NONE         0
#define VTK_KW_OPTION_LOAD                 1

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkKWMimxEditBBGroup);
vtkCxxRevisionMacro(vtkKWMimxEditBBGroup, "$Revision: 1.61 $");

//----------------------------------------------------------------------------

vtkKWMimxEditBBGroup::vtkKWMimxEditBBGroup()
{
        //this->MainFrame = NULL;
//  this->BBoxList = vtkLinkedListWrapper::New();
  this->MimxMainWindow = NULL;
  this->ObjectListComboBox = NULL;
  this->ButtonFrame = NULL;
  this->UnstructuredGridWidget = NULL;
  this->ExtractEdgeWidget = NULL;
  this->ExtractFaceWidget = NULL;
  this->ExtractCellWidget = NULL;
  this->ExtractCellWidgetHBB = NULL;
  this->RadiusEntry = NULL;
  this->MergeBBGroup = NULL;
  this->MirrorBBGroup = NULL;
 /* this->MoveButton = vtkKWRadioButton::New();
  this->AddButton = vtkKWRadioButton::New();
  this->DeleteButton = vtkKWRadioButton::New();
  this->SplitButton = vtkKWRadioButton::New();
  this->VtkInteractionButton = vtkKWRadioButton::New();
  this->EditButtonFrame = vtkKWFrameWithLabel::New();*/
  this->RadioButtonSet = NULL;
  this->AddButtonState = 0;
  this->MoveButtonState = 0;
//  this->SelectionState = 0;
  this->SplitButtonState = 0;
  this->MirrorButtonState = 0;
  this->RegularButtonState = 1;
  this->ConvertToHBBCount = 0;
  this->SplitCount = 0;
  this->AddCount = 0;
  this->DeleteCount = 0;
  this->DeleteButtonState = 0;
  this->MirrorCount = 0;
  this->MergeCount = 0;
  this->DoUndoButtonSet = NULL;
  this->SelectSubsetRadiobuttonSet = vtkKWRadioButtonSet::New();
  this->SelectCellsWidget = NULL;
  this->CancelStatus = 0;
  this->EntryFrame = NULL;
  this->defaultRadiusEntry = -1.0;
  this->defaultExtrusionLength = 1.0;
  this->defaultMergeTolerance = 0.1;

  this->MirrorFrame = vtkKWFrameWithLabel::New();
  this->TypeOfMirroring = vtkKWCheckButtonWithLabel::New();
  this->AxisSelection = vtkKWRadioButtonSet::New();
  this->MirrorPlaneWidget = NULL;
}

//----------------------------------------------------------------------------
vtkKWMimxEditBBGroup::~vtkKWMimxEditBBGroup()
{
  if(this->ObjectListComboBox)  
    this->ObjectListComboBox->Delete();
  this->RadioButtonSet->Delete();
//  this->BBoxList->Delete();
  //this->RadioButtonSet->GetWidget(0)->Delete();
  //this->RadioButtonSet->GetWidget(2)->Delete();
  //this->RadioButtonSet->GetWidget(4)->Delete();
  //this->RadioButtonSet->GetWidget(1)->Delete();
  ////this->RadioButtonSet->GetWidget(6)->Delete();
  if(this->UnstructuredGridWidget)
    this->UnstructuredGridWidget->Delete();
  if(this->ExtractEdgeWidget)
    this->ExtractEdgeWidget->Delete();
  if(this->ExtractFaceWidget)
    this->ExtractFaceWidget->Delete();
  if(this->ExtractCellWidget)
          this->ExtractCellWidget->Delete();
  if(this->RadiusEntry)
          this->RadiusEntry->Delete();
  if(this->MergeBBGroup)
          this->MergeBBGroup->Delete();
  //this->EditButtonFrame->Delete();
  this->RadioButtonSet->Delete();
  if(  this->ExtractCellWidgetHBB)
          this->ExtractCellWidgetHBB->Delete();

  if(this->DoUndoButtonSet)
          this->DoUndoButtonSet->Delete();
  if(this->SelectCellsWidget)
          this->SelectCellsWidget->Delete();
        if(this->ButtonFrame)
          this->ButtonFrame->Delete();
        if(this->EntryFrame)
                this->EntryFrame->Delete();

        this->MirrorFrame->Delete();
        this->TypeOfMirroring->Delete();
        this->AxisSelection->Delete();
        if(this->MirrorPlaneWidget)
                this->MirrorPlaneWidget->Delete();
}
//--------------------------------------------------------------------------
void vtkKWMimxEditBBGroup::SelectionChangedCallback(const char* dummy)
{
        if(this->CancelStatus)  return;
        this->AddButtonState = 1;
        this->MoveButtonState = 1;
        this->SplitButtonState = 1;
        this->DeleteButtonState = 1;
        this->MirrorButtonState = 1;
        this->ConvertToHBBButtonState = 1;
std::cout << "Selection Change Callback" << std::endl;
if( dummy != NULL )
{
        this->SelectSubsetRadiobuttonSet->SetEnabled(1);
        this->SelectFullSetCallback();
        this->defaultRadiusEntry = -1.0;
  if(this->RadioButtonSet->GetWidget(0)->GetSelectedState())
  {
    this->RadioButtonSet->GetWidget(0)->SetSelectedState(1);
//    this->EditBBMoveCellCallback();
        return;
  }

  if(this->RadioButtonSet->GetWidget(5)->GetSelectedState())
  {
          this->RadioButtonSet->GetWidget(5)->SetSelectedState(1);
//        this->EditBBMergeCallback();
          return;
  }

  if(this->RadioButtonSet->GetWidget(1)->GetSelectedState())
  {
    this->RadioButtonSet->GetWidget(1)->SetSelectedState(1);
 //   this->EditBBSplitCellCallback();
        return;
  }
  if(this->RadioButtonSet->GetWidget(2)->GetSelectedState())
  {
    std::cout << "Add Cell callback" << std::endl;
    this->RadioButtonSet->GetWidget(2)->SetSelectedState(1);
 //   this->EditBBAddCellCallback();
        return;
  }
  if(this->RadioButtonSet->GetWidget(3)->GetSelectedState())
  {
          this->RadioButtonSet->GetWidget(3)->SetSelectedState(1);
//        this->EditBBDeleteCellCallback();
          return;
  }
  if(this->RadioButtonSet->GetWidget(4)->GetSelectedState())
  {
          this->RadioButtonSet->GetWidget(4)->SetSelectedState(1);
          return;
  }
  //if(//this->RadioButtonSet->GetWidget(7)->GetSelectedState())
  //{
         // //this->RadioButtonSet->GetWidget(7)->SetSelectedState(1);
         // this->EditConvertToHBBCallback();
  //}

  //if(//this->RadioButtonSet->GetWidget(6)->GetSelectedState())
  //{
         // this->EditBBVtkInteractionCallback();
  //}
}

        vtkKWComboBox *combobox = this->ObjectListComboBox->GetWidget();
        if(!strcmp(combobox->GetValue(),""))
        {
                return;
        }
        const char *name = combobox->GetValue();
//      int num = combobox->GetValueIndex(name);
        if(this->DoUndoTree->GetItem(name)->Parent != NULL)     
                this->DoUndoButtonSet->GetWidget(0)->SetEnabled(1);
        else
                this->DoUndoButtonSet->GetWidget(0)->SetEnabled(0);

        if(this->DoUndoTree->GetItem(name)->Child != NULL)      
                this->DoUndoButtonSet->GetWidget(1)->SetEnabled(1);
        else
                this->DoUndoButtonSet->GetWidget(1)->SetEnabled(0);
}
//--------------------------------------------------------------------------
void vtkKWMimxEditBBGroup::CreateWidget()
{
  if(this->IsCreated())
  {
    vtkErrorMacro("class already created");
    return;
  }

  this->Superclass::CreateWidget();

  if(!this->ObjectListComboBox)
    this->ObjectListComboBox = vtkKWComboBoxWithLabel::New();

  this->MainFrame->SetParent(this->GetParent());
  this->MainFrame->Create();
  this->MainFrame->SetLabelText("Edit Building Block");

  this->GetApplication()->Script(
    "pack %s -side top -anchor nw -expand n -padx 2 -pady 0 -fill x", 
    this->MainFrame->GetWidgetName());

  this->ObjectListComboBox->SetParent(this->MainFrame->GetFrame());
  this->ObjectListComboBox->Create();
  this->ObjectListComboBox->SetWidth(20);
  this->ObjectListComboBox->SetLabelText("Building Block : ");
  this->ObjectListComboBox->GetWidget()->ReadOnlyOn();
  
  this->ObjectListComboBox->GetWidget()->SetCommand(this, "SelectionChangedCallback");
  this->GetApplication()->Script(
    "pack %s -side top -anchor nw -expand 0 -padx 2 -pady 6 -fill x", 
    this->ObjectListComboBox->GetWidgetName());

  this->SelectSubsetRadiobuttonSet->SetParent(this->MainFrame->GetFrame());
  this->SelectSubsetRadiobuttonSet->Create();
  this->SelectSubsetRadiobuttonSet->SetBorderWidth(2);
  this->SelectSubsetRadiobuttonSet->SetReliefToGroove();
  this->SelectSubsetRadiobuttonSet->SetMaximumNumberOfWidgetsInPackingDirection(1);
  for (int id = 0; id < 2; id++)         
          this->SelectSubsetRadiobuttonSet->AddWidget(id);

  this->SelectSubsetRadiobuttonSet->GetWidget(0)->SetCommand(this, "SelectFullSetCallback");
  this->SelectSubsetRadiobuttonSet->GetWidget(0)->SetText("Full");
//  this->SelectSubsetRadiobuttonSet->GetWidget(0)->SetImageToPredefinedIcon(vtkKWIcon::IconPlus);
  this->SelectSubsetRadiobuttonSet->GetWidget(0)->IndicatorVisibilityOff();
  this->SelectSubsetRadiobuttonSet->GetWidget(0)->SetBalloonHelpString(
          "Restore the entire building-block structure");
  this->SelectSubsetRadiobuttonSet->GetWidget(0)->SetValue("Full");
  this->SelectSubsetRadiobuttonSet->GetWidget(0)->SetCompoundModeToLeft();

//  this->SelectSubsetRadiobuttonSet->GetWidget(1)->SetText("Apply");
//  this->SelectSubsetRadiobuttonSet->GetWidget(1)->SetCommand(this, "ApplySelectSubsetCallback");
////  this->SelectSubsetRadiobuttonSet->GetWidget(1)->SetImageToPredefinedIcon(vtkKWIcon::IconPanHand);
//  this->SelectSubsetRadiobuttonSet->GetWidget(1)->IndicatorVisibilityOff();
//  this->SelectSubsetRadiobuttonSet->GetWidget(1)->SetBalloonHelpString(
//        "Apply the subset selection");
//  this->SelectSubsetRadiobuttonSet->GetWidget(1)->SetVariableName(
//        this->SelectSubsetRadiobuttonSet->GetWidget(0)->GetVariableName());
//  this->SelectSubsetRadiobuttonSet->GetWidget(1)->SetValue("Apply");
//  this->SelectSubsetRadiobuttonSet->GetWidget(1)->SetCompoundModeToLeft();

  this->SelectSubsetRadiobuttonSet->GetWidget(1)->SetText("Select");
  this->SelectSubsetRadiobuttonSet->GetWidget(1)->SetCommand(this, "SelectSubsetCallback");
  //  this->SelectSubsetRadiobuttonSet->GetWidget(1)->SetImageToPredefinedIcon(vtkKWIcon::IconPanHand);
  this->SelectSubsetRadiobuttonSet->GetWidget(1)->IndicatorVisibilityOff();
  this->SelectSubsetRadiobuttonSet->GetWidget(1)->SetBalloonHelpString(
          "Subset selection active");
  this->SelectSubsetRadiobuttonSet->GetWidget(1)->SetVariableName(
          this->SelectSubsetRadiobuttonSet->GetWidget(0)->GetVariableName());
  this->SelectSubsetRadiobuttonSet->GetWidget(1)->SetValue("Select");
  this->SelectSubsetRadiobuttonSet->GetWidget(1)->SetCompoundModeToLeft();

  //this->SelectSubsetRadiobuttonSet->GetWidget(3)->SetText("Cancel");
  //this->SelectSubsetRadiobuttonSet->GetWidget(3)->SetCommand(this, "CancelSelectSubsetCallback");
  ////  this->SelectSubsetRadiobuttonSet->GetWidget(1)->SetImageToPredefinedIcon(vtkKWIcon::IconPanHand);
  //this->SelectSubsetRadiobuttonSet->GetWidget(3)->IndicatorVisibilityOff();
  //this->SelectSubsetRadiobuttonSet->GetWidget(3)->SetBalloonHelpString(
         // "Cancel the subset selection");
  //this->SelectSubsetRadiobuttonSet->GetWidget(3)->SetVariableName(
         // this->SelectSubsetRadiobuttonSet->GetWidget(0)->GetVariableName());
  //this->SelectSubsetRadiobuttonSet->GetWidget(3)->SetValue("Cancel");
  //this->SelectSubsetRadiobuttonSet->GetWidget(3)->SetCompoundModeToLeft();

  this->GetApplication()->Script( "pack %s -side top -anchor nw -expand n -padx 2 -pady 6", 
          this->SelectSubsetRadiobuttonSet->GetWidgetName());

  if(!this->RadioButtonSet)
    this->RadioButtonSet = vtkKWCheckButtonSet::New();
  this->RadioButtonSet->SetParent(this->MainFrame->GetFrame());
  this->RadioButtonSet->Create();
  this->RadioButtonSet->SetBorderWidth(2);
  this->RadioButtonSet->SetReliefToGroove();
  this->RadioButtonSet->SetMaximumNumberOfWidgetsInPackingDirection(1);


  for (int id = 0; id < 6; id++)          this->RadioButtonSet->AddWidget(id);

  /******************* Move Button *******************/
  vtkKWIcon *moveIcon = vtkKWIcon::New();
  moveIcon->SetImage(    image_mimxMove, 
                         image_mimxMove_width, 
                         image_mimxMove_height, 
                         image_mimxMove_pixel_size); 
  this->RadioButtonSet->GetWidget(0)->SetBorderWidth(2);
  this->RadioButtonSet->GetWidget(0)->SetImageToIcon( moveIcon );
  this->RadioButtonSet->GetWidget(0)->SetSelectImageToIcon( moveIcon );
  this->RadioButtonSet->GetWidget(0)->IndicatorVisibilityOff();
  this->RadioButtonSet->GetWidget(0)->SetBalloonHelpString("Move the vertex, edge, faces of the Building Block");
//  this->RadioButtonSet->GetWidget(0)->SetVariableName(this->RadioButtonSet->GetWidget(0)->GetVariableName());
//  this->RadioButtonSet->GetWidget(0)->SetValue("Move");
  this->RadioButtonSet->GetWidget(0)->SetCompoundModeToLeft();
  this->RadioButtonSet->GetWidget(0)->SetCommand(this, "EditBBMoveCellCallback");

  /******************* Split Button *******************/
  vtkKWIcon *splitIcon = vtkKWIcon::New();
  splitIcon->SetImage(   image_mimxSplit, 
                         image_mimxSplit_width, 
                         image_mimxSplit_height, 
                         image_mimxSplit_pixel_size);
  this->RadioButtonSet->GetWidget(1)->SetBorderWidth(2);
  this->RadioButtonSet->GetWidget(1)->SetImageToIcon(splitIcon);
  this->RadioButtonSet->GetWidget(1)->SetSelectImageToIcon(splitIcon);
  this->RadioButtonSet->GetWidget(1)->IndicatorVisibilityOff();
  this->RadioButtonSet->GetWidget(1)->SetBalloonHelpString("Split the block of a Building Block Structure");
//  this->RadioButtonSet->GetWidget(1)->SetVariableName(this->RadioButtonSet->GetWidget(0)->GetVariableName());
//  this->RadioButtonSet->GetWidget(1)->SetValue("Split");
  this->RadioButtonSet->GetWidget(1)->SetCompoundModeToLeft();
  this->RadioButtonSet->GetWidget(1)->SetCommand(this, "EditBBSplitCellCallback");
  
  /******************* Add Button *******************/
  vtkKWIcon *addIcon = vtkKWIcon::New();
  addIcon->SetImage(  image_mimxAdd, 
                       image_mimxAdd_width, 
                       image_mimxAdd_height, 
                       image_mimxAdd_pixel_size);
  this->RadioButtonSet->GetWidget(2)->SetBorderWidth(2);
  this->RadioButtonSet->GetWidget(2)->SetImageToIcon( addIcon );
  this->RadioButtonSet->GetWidget(2)->SetSelectImageToIcon( addIcon );
  this->RadioButtonSet->GetWidget(2)->IndicatorVisibilityOff();
  this->RadioButtonSet->GetWidget(2)->SetBalloonHelpString("Add blocks to the Building Block Structure by selecting the faces");
//  this->RadioButtonSet->GetWidget(2)->SetValue("Add");
  this->RadioButtonSet->GetWidget(2)->SetCompoundModeToLeft();
  this->RadioButtonSet->GetWidget(2)->SetCommand(this, "EditBBAddCellCallback");

  
  /******************* Delete Button *******************/
  vtkKWIcon *deleteIcon = vtkKWIcon::New();
  deleteIcon->SetImage(  image_mimxDelete, 
                         image_mimxDelete_width, 
                         image_mimxDelete_height, 
                         image_mimxDelete_pixel_size);
  this->RadioButtonSet->GetWidget(3)->SetBorderWidth(2);
  this->RadioButtonSet->GetWidget(3)->SetReliefToRaised( );
  this->RadioButtonSet->GetWidget(3)->SetImageToIcon(deleteIcon);
  this->RadioButtonSet->GetWidget(3)->SetSelectImageToIcon(deleteIcon);
  this->RadioButtonSet->GetWidget(3)->IndicatorVisibilityOff();
  this->RadioButtonSet->GetWidget(3)->SetBalloonHelpString("Delete a block from the Building Block Structure");
//  this->RadioButtonSet->GetWidget(3)->SetVariableName(this->RadioButtonSet->GetWidget(0)->GetVariableName());
//  this->RadioButtonSet->GetWidget(3)->SetValue("Delete");
  this->RadioButtonSet->GetWidget(3)->SetCompoundModeToLeft();
  this->RadioButtonSet->GetWidget(3)->SetCommand(this, "EditBBDeleteCellCallback");

  /******************* Mirror Button *******************/
  vtkKWIcon *mirrorIcon = vtkKWIcon::New();
  mirrorIcon->SetImage(  image_mimxMirror, 
                       image_mimxMirror_width, 
                       image_mimxMirror_height, 
                       image_mimxMirror_pixel_size);                     
  this->RadioButtonSet->GetWidget(4)->SetBorderWidth(2);
  this->RadioButtonSet->GetWidget(4)->SetImageToIcon(mirrorIcon);
  this->RadioButtonSet->GetWidget(4)->SetSelectImageToIcon(mirrorIcon);
  this->RadioButtonSet->GetWidget(4)->IndicatorVisibilityOff();
  this->RadioButtonSet->GetWidget(4)->SetBalloonHelpString("Mirror a Building Block about a plane");
//  this->RadioButtonSet->GetWidget(4)->SetVariableName(this->RadioButtonSet->GetWidget(0)->GetVariableName());
//  this->RadioButtonSet->GetWidget(4)->SetValue("Mirror");
  this->RadioButtonSet->GetWidget(4)->SetCompoundModeToLeft();
  this->RadioButtonSet->GetWidget(4)->SetCommand(this, "EditBBMirrorCallback");


  /******************* Merge Button *******************/
  vtkKWIcon *mergeIcon = vtkKWIcon::New();
  mergeIcon->SetImage( image_mimxMerge, 
                       image_mimxMerge_width, 
                       image_mimxMerge_height, 
                       image_mimxMerge_pixel_size);  
  this->RadioButtonSet->GetWidget(5)->SetBorderWidth(2);
  this->RadioButtonSet->GetWidget(5)->SetImageToIcon(mergeIcon);
  this->RadioButtonSet->GetWidget(5)->SetSelectImageToIcon(mergeIcon);
  this->RadioButtonSet->GetWidget(5)->IndicatorVisibilityOff();
  this->RadioButtonSet->GetWidget(5)->SetBalloonHelpString("Merge Vertices within the specified radius");
//  this->RadioButtonSet->GetWidget(5)->SetVariableName(this->RadioButtonSet->GetWidget(0)->GetVariableName());
//  this->RadioButtonSet->GetWidget(5)->SetValue("Merge");
  this->RadioButtonSet->GetWidget(5)->SetCompoundModeToLeft();
  this->RadioButtonSet->GetWidget(5)->SetCommand(this, "EditBBMergeCallback");

  /******************* VTK Button *******************/
//  this->RadioButtonSet->GetWidget(6)->SetText("VTK");
 /* this->RadioButtonSet->GetWidget(6)->SetCommand(this, "EditBBVtkInteractionCallback");
  this->RadioButtonSet->GetWidget(6)->SetImageToPredefinedIcon(vtkKWIcon::IconNoIcon);*/
//  this->RadioButtonSet->GetWidget(6)->IndicatorVisibilityOff();
//  this->RadioButtonSet->GetWidget(6)->SetBalloonHelpString("Default VTK Interaction");
  //this->RadioButtonSet->GetWidget(6)->SetVariableName(this->RadioButtonSet->GetWidget(0)->GetVariableName());
  //this->RadioButtonSet->GetWidget(6)->SetValue("VtkInteraction");
  //this->RadioButtonSet->GetWidget(6)->SetCompoundModeToLeft();
  
 /* //this->RadioButtonSet->GetWidget(7)->SetBorderWidth(2);
  //this->RadioButtonSet->GetWidget(7)->SetText("HBB");
  //this->RadioButtonSet->GetWidget(7)->SetCommand(this, "EditConvertToHBBCallback");
  //this->RadioButtonSet->GetWidget(7)->SetImageToPredefinedIcon(vtkKWIcon::IconOrientationCubeAnnotation);
  //this->RadioButtonSet->GetWidget(7)->IndicatorVisibilityOff();
  //this->RadioButtonSet->GetWidget(7)->SetBalloonHelpString("Convert Regular Building Block to Hypercube Building Block");
  //this->RadioButtonSet->GetWidget(7)->SetVariableName(this->RadioButtonSet->GetWidget(0)->GetVariableName());
  //this->RadioButtonSet->GetWidget(7)->SetValue("HBBInteraction");
  //this->RadioButtonSet->GetWidget(7)->SetCompoundModeToLeft();
  //this->RadioButtonSet->GetWidget(7)->SetEnabled(0);
*/
  this->GetApplication()->Script( "pack %s -side top -anchor center -expand n -padx 2 -pady 6", 
          this->RadioButtonSet->GetWidgetName());
  
  if(!this->EntryFrame)
          this->EntryFrame = vtkKWFrame::New();
  this->EntryFrame->SetParent(this->GetParent());
  this->EntryFrame->Create();

  this->GetApplication()->Script(
          "pack %s -side top -anchor nw -expand n -padx 2 -pady 2 -after %s", 
          this->EntryFrame->GetWidgetName(), this->RadioButtonSet->GetWidgetName());

  if (!this->RadiusEntry)
   this->RadiusEntry = vtkKWEntryWithLabel::New();

  this->RadiusEntry->SetParent(this->EntryFrame);
  this->RadiusEntry->Create();
  this->RadiusEntry->SetWidth(4);
  this->RadiusEntry->SetLabelText("Radius : ");
  this->RadiusEntry->GetWidget()->SetValueAsDouble(1.0);
  this->RadiusEntry->GetWidget()->SetCommand(this, "RadiusChangeCallback");
  this->RadiusEntry->GetWidget()->SetRestrictValueToDouble();
  //this->GetApplication()->Script(
  // "pack %s -side top -anchor nw -expand n -padx 2 -pady 6 -before %s", 
  // this->RadioButtonSet->GetWidgetName(), this->RadiusEntry->GetWidgetName());
  //this->RadiusEntry->SetEnabled(0);

  this->MirrorFrame->SetParent(this->GetParent());
  this->MirrorFrame->Create();
  this->MirrorFrame->SetLabelText("Mirror");

  this->GetApplication()->Script(
          "pack %s -side top -anchor nw -expand n -padx 2 -pady 2 -after %s", 
          this->MirrorFrame->GetWidgetName(), this->RadioButtonSet->GetWidgetName());

  this->TypeOfMirroring->SetParent(this->MirrorFrame->GetFrame());
  this->TypeOfMirroring->Create();
  this->TypeOfMirroring->SetLabelText("About arbitrary plane");

  this->GetApplication()->Script(
          "pack %s -side top -anchor nw -expand y -padx 2 -pady 6", 
          this->TypeOfMirroring->GetWidgetName());
  this->TypeOfMirroring->GetWidget()->SetEnabled(0);

  this->AxisSelection->SetParent(this->MirrorFrame->GetFrame());
  this->AxisSelection->Create();
  this->AxisSelection->SetBorderWidth(2);
  this->AxisSelection->SetReliefToGroove();

  vtkKWRadioButton *rb;
  rb = this->AxisSelection->AddWidget(0);
  rb->SetText("About XY Plane");
  rb = this->AxisSelection->AddWidget(1);
  rb->SetText("About XZ Plane");
  rb = this->AxisSelection->AddWidget(2);
  rb->SetText("About YZ Plane");

  this->AxisSelection->GetWidget(0)->SetCommand(this, "PlaceMirroringPlaneAboutZ");
  this->AxisSelection->GetWidget(1)->SetCommand(this, "PlaceMirroringPlaneAboutY");
  this->AxisSelection->GetWidget(2)->SetCommand(this, "PlaceMirroringPlaneAboutX");

  this->GetApplication()->Script(
          "pack %s -side top -anchor nw -expand n -padx 2 -pady 6", 
          this->AxisSelection->GetWidgetName());
  this->MirrorFrame->Unpack();

  if (!this->ButtonFrame)
    this->ButtonFrame = vtkKWFrame::New();
  this->ButtonFrame->SetParent( this->MainFrame->GetFrame() );
  this->ButtonFrame->Create();
  this->GetApplication()->Script("pack %s -side top -anchor nw -expand n -fill x -padx 2 -pady 2",
              this->ButtonFrame->GetWidgetName() ); 
                          
        this->ApplyButton->SetParent(this->ButtonFrame);
        this->ApplyButton->Create();
        this->ApplyButton->SetText("Apply");
        this->ApplyButton->SetCommand(this, "EditBBApplyCallback");
        this->GetApplication()->Script(
                "pack %s -side left -anchor nw -expand y -padx 5 -pady 6", 
                this->ApplyButton->GetWidgetName());

  this->CancelButton->SetParent(this->ButtonFrame);
  this->CancelButton->Create();
  this->CancelButton->SetText("Cancel");
  this->CancelButton->SetCommand(this, "EditBBCancelCallback");
  this->GetApplication()->Script(
    "pack %s -side right -anchor ne -expand n -padx 5 -pady 6", 
    this->CancelButton->GetWidgetName());

  if(!this->DoUndoButtonSet)
          this->DoUndoButtonSet = vtkKWPushButtonSet::New();

  this->DoUndoButtonSet->SetParent(this->MainFrame->GetFrame());
  this->DoUndoButtonSet->Create();
  this->DoUndoButtonSet->SetBorderWidth(2);
  this->DoUndoButtonSet->SetReliefToGroove();
  this->DoUndoButtonSet->SetMaximumNumberOfWidgetsInPackingDirection(1);
  for (int id = 0; id < 2; id++)          this->DoUndoButtonSet->AddWidget(id);
  
  
  /******************* Undo Button *******************/
  vtkKWIcon *undoIcon = vtkKWIcon::New();
  undoIcon->SetImage(  image_mimxUndo, 
                       image_mimxUndo_width, 
                       image_mimxUndo_height, 
                       image_mimxUndo_pixel_size);
  this->DoUndoButtonSet->GetWidget(0)->SetImageToIcon( undoIcon );
  this->DoUndoButtonSet->GetWidget(0)->SetBalloonHelpString("Undo building-block editing operations");
  this->DoUndoButtonSet->GetWidget(0)->SetCompoundModeToLeft();
  this->DoUndoButtonSet->GetWidget(0)->SetEnabled(0);
  this->DoUndoButtonSet->GetWidget(0)->SetCommand(this, "UndoBBCallback");

  /******************* Redo Button *******************/
  vtkKWIcon *redoIcon = vtkKWIcon::New();
  redoIcon->SetImage(  image_mimxRedo, 
                       image_mimxRedo_width, 
                       image_mimxRedo_height, 
                       image_mimxRedo_pixel_size);
  this->DoUndoButtonSet->GetWidget(1)->SetImageToIcon( redoIcon );
  this->DoUndoButtonSet->GetWidget(1)->SetBalloonHelpString("Redo building-block editing operations");
  this->DoUndoButtonSet->GetWidget(1)->SetCompoundModeToRight();
  this->DoUndoButtonSet->GetWidget(1)->SetEnabled(0);
  this->DoUndoButtonSet->GetWidget(1)->SetCommand(this, "DoBBCallback");

  this->GetApplication()->Script( "pack %s -side top -anchor center -padx 2 -pady 6", 
          this->DoUndoButtonSet->GetWidgetName());
          
  //this->GetApplication()->Script(
         // "pack forget %s", this->RadiusEntry->GetWidgetName());
  this->DoUndoButtonSet->SetEnabled(0);
  this->SelectSubsetRadiobuttonSet->SetEnabled(0);
}
//----------------------------------------------------------------------------
void vtkKWMimxEditBBGroup::Update()
{
        this->UpdateEnableState();
}
//---------------------------------------------------------------------------
void vtkKWMimxEditBBGroup::UpdateEnableState()
{
        this->UpdateObjectLists();
        this->Superclass::UpdateEnableState();
}
//----------------------------------------------------------------------------
int vtkKWMimxEditBBGroup::EditBBApplyCallback()
{
        vtkMimxErrorCallback *callback = this->GetMimxMainWindow()->GetErrorCallback();
        callback->SetState(0);
std::cout << "In Apply Callback" << std::endl;

// Merge Callback
if(this->RadioButtonSet->GetWidget(5)->GetSelectedState())
{
                vtkKWComboBox *combobox = this->ObjectListComboBox->GetWidget();
                const char *name = combobox->GetValue();

                int num = combobox->GetValueIndex(name);
                if(num < 0 || num > combobox->GetNumberOfValues()-1)
                {
                        callback->ErrorMessage("Choose valid Building-block structure");
                        combobox->SetValue("");
                        return 0;
                }

                vtkMimxUnstructuredGridActor *ugridactor = vtkMimxUnstructuredGridActor::
                        SafeDownCast(this->BBoxList->GetItem(combobox->GetValueIndex(name)));
                vtkUnstructuredGrid *ugrid = ugridactor->GetDataSet();

                vtkUnstructuredGrid *Ugrid = vtkUnstructuredGrid::New();
                vtkMergeCells* mergecells = vtkMergeCells::New();
                mergecells->SetUnstructuredGrid(Ugrid);
                double tol = this->RadiusEntry->GetWidget()->GetValueAsDouble();
                mergecells->SetPointMergeTolerance(tol);
                mergecells->SetTotalNumberOfDataSets(1);
                mergecells->SetTotalNumberOfCells(ugrid->GetNumberOfCells());
                mergecells->SetTotalNumberOfPoints(ugrid->GetNumberOfPoints());
                //              mergecells->SetMergeDuplicatePoints(1);
                mergecells->AddObserver(vtkCommand::ErrorEvent, callback, 1.0);
                mergecells->MergeDataSet(ugrid);
                mergecells->Finish();
                mergecells->RemoveObserver(callback);

                if (callback->GetState())
                {
                        mergecells->RemoveObserver(callback);
                        mergecells->Delete();
                        //this->RadioButtonSet->GetWidget(6)->SelectedStateOn();
                        Ugrid->Delete();
                        return 0;
                }
                else
                {
                        this->AddEditedBB(num, Ugrid, "Merge_", this->MergeCount);
                        Ugrid->Delete();
                        this->GetMimxMainWindow()->GetRenderWidget()->RemoveViewProp(
                                this->BBoxList->GetItem(this->BBoxList->GetNumberOfItems()-1)->GetActor());
                        mergecells->RemoveObserver(callback);

                        vtkUnstructuredGrid *ugrid = vtkMimxUnstructuredGridActor::SafeDownCast(
                                this->BBoxList->GetItem(this->BBoxList->GetNumberOfItems()-1))->GetDataSet();
                        vtkActor *actor = this->BBoxList->GetItem(
                                this->BBoxList->GetNumberOfItems()-1)->GetActor();

                        vtkMimxMapOriginalCellAndPointIds *mapcellspoints = 
                                vtkMimxMapOriginalCellAndPointIds::New();
                        mapcellspoints->SetCompleteMesh(ugrid);
                        mapcellspoints->SetPartialMesh(ugrid);
                        mapcellspoints->Update();
                        mapcellspoints->Delete();

                        this->SelectCellsWidget->SetInputAndCurrentSelectedMesh(
                                ugrid, ugrid);
                        this->GetMimxMainWindow()->GetRenderWidget()->GetRenderer()->AddActor(
                                this->SelectCellsWidget->GetCurrentSelectedSubsetActor());
                        this->GetMimxMainWindow()->GetRenderWidget()->Render();
                        mergecells->Delete();
                        this->RadioButtonSet->GetWidget(5)->SetSelectedState(0);          
                        return 1;
                }
                
        //this->RadioButtonSet->GetWidget(6)->SelectedStateOn();
        return 0;
}


  // Split Cell Callback
  if(this->RadioButtonSet->GetWidget(1)->GetSelectedState())
  {
    if (this->ExtractEdgeWidget)
    {
      vtkKWComboBox *combobox = this->ObjectListComboBox->GetWidget();
      const char *name = combobox->GetValue();

          int num = combobox->GetValueIndex(name);
          if(num < 0 || num > combobox->GetNumberOfValues()-1)
          {
                  callback->ErrorMessage("Choose valid Building-block structure");
                  combobox->SetValue("");
                  return 0;
          }

          vtkMimxUnstructuredGridActor *ugridactor = vtkMimxUnstructuredGridActor::
                  SafeDownCast(this->BBoxList->GetItem(combobox->GetValueIndex(name)));
      vtkUnstructuredGrid *ugrid = ugridactor->GetDataSet();

          vtkIdList *edgepoints = this->ExtractEdgeWidget->GetEdgePoints();
          if(edgepoints->GetNumberOfIds() != 2)
          {
                  callback->ErrorMessage("Invalid edge selection");
                  return 0;
          }

          vtkIdList *edgepointscomplete = this->ExtractEdgeWidget->GetEdgePointsCompleteGrid();
          if(edgepointscomplete->GetNumberOfIds() != 2)
          {
                  callback->ErrorMessage("Invalid edge selection");
                  return 0;
          }

          vtkMimxSplitUnstructuredHexahedronGridCell *splitpartial = 
                  vtkMimxSplitUnstructuredHexahedronGridCell::New();
          splitpartial->SetInput(this->SelectCellsWidget->GetCurrentSelectedSubset());

          splitpartial->SetIdList(edgepoints);
          splitpartial->AddObserver(vtkCommand::ErrorEvent, callback, 1.0);
          splitpartial->Update();

          if (callback->GetState())
          {
                  splitpartial->RemoveObserver(callback);
                  splitpartial->Delete();
                  //this->RadioButtonSet->GetWidget(6)->SelectedStateOn();
                  return 0;
          }
          else
          {
                  this->ExtractEdgeWidget->SetEnabled(0);
                  splitpartial->RemoveObserver(callback);
          }

          vtkMimxSplitUnstructuredHexahedronGridCell *splitcomplete = 
                  vtkMimxSplitUnstructuredHexahedronGridCell::New();
          splitcomplete->SetInput(ugrid);
          splitcomplete->SetIdList(edgepointscomplete);
          splitcomplete->AddObserver(vtkCommand::ErrorEvent, callback, 1.0);
          splitcomplete->Update();

          if (callback->GetState())
          {
                  splitcomplete->RemoveObserver(callback);
                  splitcomplete->Delete();
                  //this->RadioButtonSet->GetWidget(6)->SelectedStateOn();
                  return 0;
          }
          else
          {
                  this->ExtractEdgeWidget->SetEnabled(0);
                  this->AddEditedBB(num, splitcomplete->GetOutput(), "Split_", this->SplitCount);
                  this->GetMimxMainWindow()->GetRenderWidget()->RemoveViewProp(
                          this->BBoxList->GetItem(this->BBoxList->GetNumberOfItems()-1)->GetActor());
                  splitcomplete->RemoveObserver(callback);

                  vtkUnstructuredGrid *ugrid = vtkMimxUnstructuredGridActor::SafeDownCast(
                          this->BBoxList->GetItem(this->BBoxList->GetNumberOfItems()-1))->GetDataSet();
                  vtkActor *actor = this->BBoxList->GetItem(
                          this->BBoxList->GetNumberOfItems()-1)->GetActor();

                  vtkMimxMapOriginalCellAndPointIds *mapcellspoints = 
                          vtkMimxMapOriginalCellAndPointIds::New();
                  mapcellspoints->SetCompleteMesh(splitcomplete->GetOutput());
                  mapcellspoints->SetPartialMesh(splitpartial->GetOutput());
                  mapcellspoints->Update();
                  this->SelectCellsWidget->SetInputAndCurrentSelectedMesh(
                          ugrid, splitpartial->GetOutput());
                  this->GetMimxMainWindow()->GetRenderWidget()->GetRenderer()->AddActor(
                          this->SelectCellsWidget->GetCurrentSelectedSubsetActor());
                  this->GetMimxMainWindow()->GetRenderWidget()->Render();
                  splitcomplete->Delete();
                  splitpartial->Delete();
                  mapcellspoints->Delete();
                  
                  this->GetMimxMainWindow()->SetStatusText("Split Building Block");
                  this->RadioButtonSet->GetWidget(1)->SelectedStateOff();         
                  return 1;
          }

     }
        //this->RadioButtonSet->GetWidget(6)->SelectedStateOn();
        return 0;
  }
  
  // Add Cell Callback
  if(this->RadioButtonSet->GetWidget(2)->GetSelectedState())
  {
          if(this->ExtractFaceWidget)
          {
                  vtkKWComboBox *combobox = this->ObjectListComboBox->GetWidget();
                  const char *name = combobox->GetValue();
                  int num = combobox->GetValueIndex(name);
      
                  if(num < 0 || num > combobox->GetNumberOfValues()-1)
                  {
                          callback->ErrorMessage("No Building Block structure was selected");
                          combobox->SetValue("");
                          return 0;
                  }
                  
                  vtkMimxUnstructuredGridActor *ugridactorprev = vtkMimxUnstructuredGridActor::
                          SafeDownCast(this->BBoxList->GetItem(combobox->GetValueIndex(name)));

                  vtkUnstructuredGrid *ugrid = vtkMimxUnstructuredGridActor::SafeDownCast(
                          this->BBoxList->GetItem(combobox->GetValueIndex(name)))->GetDataSet();

                  vtkIntArray *intarray = vtkIntArray::SafeDownCast(
                          ugrid->GetCellData()->GetArray("Mesh_Seed"));

                  // for partial mesh
                  vtkMimxExtractSurface *extractpartial = vtkMimxExtractSurface::New();
                  extractpartial->AddObserver(vtkCommand::ErrorEvent, callback, 1.0);
                  extractpartial->SetInput(this->SelectCellsWidget->GetCurrentSelectedSubset());
                  extractpartial->SetCellIdList(this->ExtractFaceWidget->GetPickedCellList());
                  extractpartial->SetFaceIdList(this->ExtractFaceWidget->GetPickedFaceList());
                  extractpartial->Update();
                
                  if (callback->GetState())
                  {
                          extractpartial->RemoveObserver(callback);
                          extractpartial->Delete();
                          //this->RadioButtonSet->GetWidget(6)->SelectedStateOn();
                          return 0;
                  }
                  else
                  {
                          extractpartial->RemoveObserver(callback);
                  }

                  this->defaultExtrusionLength = this->RadiusEntry->GetWidget()->GetValueAsDouble();
//<<<<<<< vtkKWMimxEditBBGroup.cxx
                  vtkMimxExtrudePolyData *extrudepartial = vtkMimxExtrudePolyData::New();
                  extrudepartial->SetExtrusionLength(this->RadiusEntry->GetWidget()->GetValueAsDouble());
                  extrudepartial->SetInput(extractpartial->GetOutput());
                  extrudepartial->Update();

                  //vtkUnstructuredGridWriter *writer2 = vtkUnstructuredGridWriter::New();
                  //writer2->SetInput(extrudepartial->GetOutput());
                  //writer2->SetFileName("Extrude.vtk");
                  //writer2->Write();

                  vtkAppendFilter *AppendPartial = vtkAppendFilter::New();
                  AppendPartial->AddInput(this->SelectCellsWidget->GetCurrentSelectedSubset());
                  AppendPartial->AddInput(extrudepartial->GetOutput());
                  AppendPartial->Update();

                  vtkMergeCells *mergecellspartial = vtkMergeCells::New();
                  vtkUnstructuredGrid *ugridpartial = vtkUnstructuredGrid::New();

                  mergecellspartial->SetUnstructuredGrid(ugridpartial);
                  mergecellspartial->SetTotalNumberOfDataSets(1);
                  mergecellspartial->SetTotalNumberOfCells(1000);
                  mergecellspartial->SetTotalNumberOfPoints(1000);
                  mergecellspartial->MergeDataSet(AppendPartial->GetOutput());
                  mergecellspartial->Finish();
                  mergecellspartial->Delete();
                  extrudepartial->Delete();
                  extractpartial->Delete();
                  AppendPartial->Delete();
                  // for complete mesh
                  vtkMimxExtractSurface *extractcomplete = vtkMimxExtractSurface::New();
                  extractcomplete->AddObserver(vtkCommand::ErrorEvent, callback, 1.0);
                  extractcomplete->SetInput(ugrid);
                  extractcomplete->SetCellIdList(this->ExtractFaceWidget->GetCompletePickedCellList());
                  extractcomplete->SetFaceIdList(this->ExtractFaceWidget->GetCompletePickedFaceList());
                  extractcomplete->Update();

                  if (callback->GetState())
                  {
                          extractcomplete->RemoveObserver(callback);
                          extractcomplete->Delete();
                          //this->RadioButtonSet->GetWidget(6)->SelectedStateOn();
                          return 0;
                  }
                  else
                  {
                          this->ExtractFaceWidget->SetEnabled(0);
                          extractcomplete->RemoveObserver(callback);
                  }

                  this->defaultExtrusionLength = this->RadiusEntry->GetWidget()->GetValueAsDouble();
                  vtkMimxExtrudePolyData *extrudecomplete = vtkMimxExtrudePolyData::New();
                  extrudecomplete->SetExtrusionLength(this->RadiusEntry->GetWidget()->GetValueAsDouble());
                  extrudecomplete->SetInput(extractcomplete->GetOutput());
                  extrudecomplete->Update();

                  vtkAppendFilter *AppendComplete = vtkAppendFilter::New();
                  AppendComplete->AddInput(ugrid);
                  AppendComplete->AddInput(extrudecomplete->GetOutput());
                  AppendComplete->Update();

                  vtkMergeCells *mergecellscomplete = vtkMergeCells::New();
                  vtkUnstructuredGrid *ugridcomplete = vtkUnstructuredGrid::New();

                  mergecellscomplete->SetUnstructuredGrid(ugridcomplete);
                  mergecellscomplete->SetTotalNumberOfDataSets(1);
                  mergecellscomplete->SetTotalNumberOfCells(1000);
                  mergecellscomplete->SetTotalNumberOfPoints(1000);
                  mergecellscomplete->MergeDataSet(AppendComplete->GetOutput());
                  mergecellscomplete->Finish();
                  mergecellscomplete->Delete();
                  extrudecomplete->Delete();
                  extractcomplete->Delete();
                  AppendComplete->Delete();

                  this->AddEditedBB(num, ugridcomplete, "Add_", this->AddCount);
                  this->GetMimxMainWindow()->GetRenderWidget()->RemoveViewProp(
                          this->BBoxList->GetItem(this->BBoxList->GetNumberOfItems()-1)->GetActor());

                  vtkMimxUnstructuredGridActor *ugridactor = vtkMimxUnstructuredGridActor::SafeDownCast(
                          this->BBoxList->GetItem(this->BBoxList->GetNumberOfItems()-1));
                  // if mesh seeds are present then copy the mesh seeds of the previous mesh and set the
                  // mesh seeds to be one for the newly added building blocks

                  ugrid = vtkMimxUnstructuredGridActor::SafeDownCast(
                          this->BBoxList->GetItem(this->BBoxList->GetNumberOfItems()-1))->GetDataSet();


                  ugridactor = vtkMimxUnstructuredGridActor::SafeDownCast(
                          this->BBoxList->GetItem(this->BBoxList->GetNumberOfItems()-1));

                  vtkActor *actor = this->BBoxList->GetItem(
                          this->BBoxList->GetNumberOfItems()-1)->GetActor();

                  vtkMimxMapOriginalCellAndPointIds *mapcellspoints = 
                          vtkMimxMapOriginalCellAndPointIds::New();
                  mapcellspoints->SetCompleteMesh(ugridcomplete);
                  mapcellspoints->SetPartialMesh(ugridpartial);
                  mapcellspoints->Update();
                  this->SelectCellsWidget->SetInputAndCurrentSelectedMesh(
                          ugrid, ugridpartial);
                  this->GetMimxMainWindow()->GetRenderWidget()->GetRenderer()->AddActor(
                          this->SelectCellsWidget->GetCurrentSelectedSubsetActor());
                  this->GetMimxMainWindow()->GetRenderWidget()->Render();

                  if(intarray)
                  {
                          ugridactor->MeshSeedFromAverageElementLength(1.0, 1.0, 1.0);
                          
                           int dim[3];
                           dim[0] =2; dim[1] =2; dim[2] =2;
                           int i,j;
                                for (i=0; i<ugrid->GetNumberOfCells(); i++)
                                {
                                        for (j=0; j<3; j++)
                                        {
                                                ugridactor->ChangeMeshSeed(i,j, dim[j]);
                                        }
                                }
                                
                                for (i=0; i<intarray->GetNumberOfTuples(); i++)
                                {
                                        intarray->GetTupleValue(i, dim);
                                        for (j=0; j<3; j++)
                                        {
                                                ugridactor->ChangeMeshSeed(i,j, dim[j]);
                                        }
                                }
                  }

                  mapcellspoints->Delete();
                  ugridcomplete->Delete();
                  ugridpartial->Delete();                 
                  this->GetMimxMainWindow()->SetStatusText("Added Building Block");
                  this->RadioButtonSet->GetWidget(2)->SetSelectedState(0);        

                  return 1;
        }

          return 0;     
  }

  // Delete Cell Callback
  if(this->RadioButtonSet->GetWidget(3)->GetSelectedState())
  {
          if(this->ExtractCellWidget)
          {
                  if(this->ExtractCellWidget->GetPickedCells()->GetNumberOfIds() > 0)
                  {

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

                  vtkMimxDeleteUnstructuredHexahedronGridCell *deletepartial = 
                          vtkMimxDeleteUnstructuredHexahedronGridCell::New();
                  deletepartial->SetInput(this->SelectCellsWidget->GetCurrentSelectedSubset());

                  vtkIdList *idlist= this->ExtractCellWidget->GetPickedCells();
                  deletepartial->SetCellNum(idlist->GetId(0));
                  deletepartial->AddObserver(vtkCommand::ErrorEvent, callback, 1.0);
                  deletepartial->Update();

                  if (callback->GetState())
                  {
                          deletepartial->RemoveObserver(callback);
                          deletepartial->Delete();
                          //this->RadioButtonSet->GetWidget(6)->SelectedStateOn();
                          return 0;
                  }
                  else
                  {
                          this->ExtractCellWidget->SetEnabled(0);
                          deletepartial->RemoveObserver(callback);
                  }

                  vtkMimxDeleteUnstructuredHexahedronGridCell *deletecomplete = 
                          vtkMimxDeleteUnstructuredHexahedronGridCell::New();
                  deletecomplete->SetInput(ugrid);

//                idlist= this->ExtractCellWidget->GetPickedCellsCompleteGrid();
                  deletecomplete->SetCellNum(idlist->GetId(0));
                  deletecomplete->AddObserver(vtkCommand::ErrorEvent, callback, 1.0);
                  deletecomplete->Update();

                  if (callback->GetState())
                  {
                          deletecomplete->RemoveObserver(callback);
                          deletecomplete->Delete();
                          //this->RadioButtonSet->GetWidget(6)->SelectedStateOn();
                          return 0;
                  }
                  else
                  {
                          this->ExtractCellWidget->SetEnabled(0);
                          this->AddEditedBB(num, deletecomplete->GetOutput(), "Delete_", this->DeleteCount);
                          this->GetMimxMainWindow()->GetRenderWidget()->RemoveViewProp(
                                  this->BBoxList->GetItem(this->BBoxList->GetNumberOfItems()-1)->GetActor());
                          deletecomplete->RemoveObserver(callback);

                          vtkUnstructuredGrid *ugrid = vtkMimxUnstructuredGridActor::SafeDownCast(
                                  this->BBoxList->GetItem(this->BBoxList->GetNumberOfItems()-1))->GetDataSet();
                          vtkActor *actor = this->BBoxList->GetItem(
                                  this->BBoxList->GetNumberOfItems()-1)->GetActor();

                          vtkMimxMapOriginalCellAndPointIds *mapcellspoints = 
                                  vtkMimxMapOriginalCellAndPointIds::New();
                          mapcellspoints->SetCompleteMesh(deletecomplete->GetOutput());
                          mapcellspoints->SetPartialMesh(deletepartial->GetOutput());
                          mapcellspoints->Update();
                          this->SelectCellsWidget->SetInputAndCurrentSelectedMesh(
                                  ugrid, deletepartial->GetOutput());
                          this->GetMimxMainWindow()->GetRenderWidget()->GetRenderer()->AddActor(
                                  this->SelectCellsWidget->GetCurrentSelectedSubsetActor());
                          this->GetMimxMainWindow()->GetRenderWidget()->Render();
                          deletecomplete->Delete();
                          deletepartial->Delete();
                          mapcellspoints->Delete();
                          this->GetMimxMainWindow()->SetStatusText("Deleted Building Block Cell");
                          this->RadioButtonSet->GetWidget(3)->SetSelectedState(0);        
                          return 1;
                  }
          }
          }
          //this->RadioButtonSet->GetWidget(6)->SelectedStateOn();
          return 0;     
  }

  if(this->RadioButtonSet->GetWidget(4)->GetSelectedState())
  {
          if(this->MirrorPlaneWidget)
          {
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
                  vtkMimxMirrorUnstructuredHexahedronGridCell *mirror = 
                          vtkMimxMirrorUnstructuredHexahedronGridCell::New();
                  mirror->SetInput(ugrid);
                  vtkPlane *Plane = vtkPlane::New();
                  this->MirrorPlaneWidget->GetPlane(Plane);
                  mirror->SetMirrorPlane(Plane);
                  mirror->AddObserver(vtkCommand::ErrorEvent, callback, 1.0);
                  mirror->Update();
                  vtkIdType objnum = combobox->GetValueIndex(name);
                  if (!callback->GetState())
                  {
                          this->AddEditedBB(num, mirror->GetOutput(), "Mirror_", this->MirrorCount);
                          mirror->RemoveObserver(callback);
                          mirror->Delete();
                          Plane->Delete();
                          vtkUnstructuredGrid *ugrid = vtkMimxUnstructuredGridActor::SafeDownCast(
                                  this->BBoxList->GetItem(this->BBoxList->GetNumberOfItems()-1))->GetDataSet();
                          this->SelectCellsWidget->SetInput(ugrid);
                          this->GetMimxMainWindow()->GetRenderWidget()->AddViewProp(
                          this->SelectCellsWidget->GetCurrentSelectedSubsetActor());
                          this->GetMimxMainWindow()->GetRenderWidget()->RemoveViewProp(
                                  this->BBoxList->GetItem(this->BBoxList->GetNumberOfItems()-1)->GetActor());
                          this->RadioButtonSet->GetWidget(4)->SetSelectedState(0);
                          this->GetMimxMainWindow()->SetStatusText("Mirrored Building Block");
                          this->MirrorPlaneWidget->SetEnabled(0);
                          return 1;
                  }
                  return 0;
          }
  }

  // Convert to Hypercube Callback
  //if(//this->RadioButtonSet->GetWidget(7)->GetSelectedState())
  //{
         // if(this->ExtractCellWidgetHBB)
         // {
                //  if(this->ExtractCellWidgetHBB->GetPickedCells()->GetNumberOfIds() > 0)
                //  {
                //        vtkKWComboBox *combobox = this->ObjectListComboBox->GetWidget();
                //        const char *name = combobox->GetValue();

                //        int num = combobox->GetValueIndex(name);
                //        if(num < 0 || num > combobox->GetNumberOfValues()-1)
                //        {
                //                callback->ErrorMessage("Choose valid Building-block structure");
                //                combobox->SetValue("");
                //                return 0;
                //        }

                //        vtkUnstructuredGrid *ugrid = vtkMimxUnstructuredGridActor::SafeDownCast(
                //                this->BBoxList->GetItem(combobox->GetValueIndex(name)))->GetDataSet();
                //        vtkMimxSubdivideBoundingbox * subdivide = vtkMimxSubdivideBoundingbox::New();
                //        subdivide->SetInput(ugrid);
                //        subdivide->SetCellNum(this->ExtractCellWidgetHBB->GetPickedCells()->GetId(0));
                //        subdivide->AddObserver(vtkCommand::ErrorEvent, callback, 1.0);
                //        subdivide->Update();

                //        if (!callback->GetState())
                //        {
                //                this->ExtractCellWidgetHBB->SetEnabled(0);
                //                this->AddEditedBB(num, subdivide->GetOutput(), "ConvertToHBB_", this->ConvertToHBBCount);
  //        subdivide->RemoveObserver(callback);
  //        subdivide->Delete();
  //        return 1;
                //        }
                //        subdivide->RemoveObserver(callback);
                //        subdivide->Delete();
                //        //this->RadioButtonSet->GetWidget(6)->SelectedStateOn();
                //        return 0;
                //  }
         // }
         // //this->RadioButtonSet->GetWidget(6)->SelectedStateOn();
         // return 0;   
  //}

  //
 /* if(!this->RadioButtonSet->GetWidget(1)->GetSelectedState())
  {
        callback->ErrorMessage("Choose an editing button");
  }*/
  return 0;
}
//----------------------------------------------------------------------------
void vtkKWMimxEditBBGroup::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
//----------------------------------------------------------------------------
void vtkKWMimxEditBBGroup::EditBBCancelCallback()
{

        this->CancelStatus = 1;
        this->DeselectAllButtons();
        this->GetApplication()->Script("pack forget %s", this->MainFrame->GetWidgetName());
        this->MenuGroup->SetMenuButtonsEnabled(1);
        if(this->ExtractCellWidget)
                this->ExtractCellWidget->Delete();
        this->ExtractCellWidget = NULL;

        if(this->ExtractFaceWidget)
                this->ExtractFaceWidget->Delete();
        this->ExtractFaceWidget = NULL;

        if(this->ExtractEdgeWidget)
                this->ExtractEdgeWidget->Delete();
        this->ExtractEdgeWidget = NULL;
        
        if(this->ExtractCellWidgetHBB)
                this->ExtractCellWidgetHBB->Delete();
        this->ExtractCellWidgetHBB = NULL;

        if(this->UnstructuredGridWidget)
                this->UnstructuredGridWidget->Delete();
        this->UnstructuredGridWidget = NULL;

        if(this->SelectCellsWidget)
        {
                if(this->SelectCellsWidget->GetCurrentSelectedSubsetActor())
                {
                        this->GetMimxMainWindow()->GetRenderWidget()->RemoveViewProp(
                                this->SelectCellsWidget->GetCurrentSelectedSubsetActor());
                }
                this->SelectCellsWidget->SetEnabled(0);
                this->SelectCellsWidget->Delete();
                this->SelectCellsWidget = NULL;
                this->GetMimxMainWindow()->GetRenderWidget()->Render();
        }
        
        this->GetMimxMainWindow()->GetMainUserInterfacePanel()->GetMimxMainNotebook()->SetEnabled(1);

        if(!strcmp(this->ObjectListComboBox->GetWidget()->GetValue(),""))
        {
                this->CancelStatus = 0;
    return;
        }

        vtkKWComboBox *combobox = this->ObjectListComboBox->GetWidget();
        const char *name = combobox->GetValue();
        int num = combobox->GetValueIndex(name);
        if(num < 0 || num > combobox->GetNumberOfValues()-1)
        {
                this->CancelStatus = 0;
    return;
        }

        this->GetMimxMainWindow()->GetRenderWidget()->AddViewProp(
                this->BBoxList->GetItem(combobox->GetValueIndex(name))->GetActor());
        this->GetMimxMainWindow()->GetRenderWidget()->Render();
        this->GetMimxMainWindow()->GetViewProperties()->UpdateVisibility();


        this->CancelStatus = 0;
        
}
//----------------------------------------------------------------------------
void vtkKWMimxEditBBGroup::EditBBDeleteCellCallback(int Mode)
{
        if(!Mode)
        {
                if(this->ExtractCellWidget)
                {
                        if(this->ExtractCellWidget->GetEnabled())
                        {
                                this->ExtractCellWidget->SetEnabled(0);
                        }
                }
                this->EntryFrame->Unpack();
                this->SetDoUndoButtonSelectSubsetButton();
                if(!this->SelectCellsWidget)    return;
                if(!this->SelectCellsWidget->GetCurrentSelectedSubsetActor()) return;
                this->GetMimxMainWindow()->GetRenderWidget()->AddViewProp(
                        this->SelectCellsWidget->GetCurrentSelectedSubsetActor());
                this->GetMimxMainWindow()->GetRenderWidget()->Render();
                return;
        }

        vtkMimxErrorCallback *callback = this->GetMimxMainWindow()->GetErrorCallback();

        this->CancelStatus = 1;
        this->EntryFrame->Unpack();
        this->CancelStatus = 0;
//      this->RadiusEntry->SetEnabled(0);
        this->SelectSubsetRadiobuttonSet->SetEnabled(0);
        if(!strcmp(this->ObjectListComboBox->GetWidget()->GetValue(),""))
        {
                if(!this->CancelStatus)
                {
                        callback->ErrorMessage("Building Block selection required");
                        this->DeselectAllButtons();
                }
        }
        else
        {
                vtkKWComboBox *combobox = this->ObjectListComboBox->GetWidget();
                const char *name = combobox->GetValue();
                int num = combobox->GetValueIndex(name);
                if(num < 0 || num > combobox->GetNumberOfValues()-1)
                {
                        callback->ErrorMessage("Choose valid Building-block structure");
                        combobox->SetValue("");
                        this->DeselectAllButtons();
                        return;
                }
                vtkUnstructuredGrid *ugrid = vtkMimxUnstructuredGridActor::SafeDownCast(this->BBoxList
                        ->GetItem(combobox->GetValueIndex(name)))->GetDataSet();

                if(!this->SelectCellsWidget)
                {
                        this->SelectCellsWidget = vtkMimxSelectCellsWidget::New();
                        //this->SelectCellsWidget->SetInputActor(this->BBoxList
                        //      ->GetItem(combobox->GetValueIndex(name))->GetActor());
                        this->SelectCellsWidget->SetInput(ugrid);
                        this->SelectCellsWidget->SetInteractor(
                                this->GetMimxMainWindow()->GetRenderWidget()->GetRenderWindowInteractor());
                        this->SelectCellsWidget->Initialize();
                }

                //if(this->DeleteButtonState)
                //{
                                if(this->ExtractCellWidget)
                        {
                                if(this->ExtractCellWidget->GetEnabled())
                                {
                                        this->ExtractCellWidget->SetEnabled(0);
                                }
                                this->ExtractCellWidget->Delete();
                                this->ExtractCellWidget = NULL;
                        }
                        this->ExtractCellWidget = vtkMimxExtractCellWidget::New();
                        this->ExtractCellWidget->SetInteractor(this->GetMimxMainWindow()->GetRenderWidget()
                                ->GetRenderWindowInteractor());
                        this->ExtractCellWidget->SetInput(this->SelectCellsWidget->GetCurrentSelectedSubset());
                        //this->ExtractCellWidget->SetInputActor(this->BBoxList
                        //      ->GetItem(combobox->GetValueIndex(name))->GetActor());
                        this->ExtractCellWidget->SetEnabled(
                                this->RadioButtonSet->GetWidget(3)->GetSelectedState());
                        this->DeleteButtonState = 0;
                //}
                //else
                //{
                        //if(!this->ExtractCellWidget)
                        //{
                        //      vtkUnstructuredGrid *ugrid = vtkMimxUnstructuredGridActor::SafeDownCast(
                        //              this->BBoxList->GetItem(combobox->GetValueIndex(name)))->GetDataSet();
                        //      this->ExtractCellWidget = vtkMimxExtractCellWidget::New();
                        //      this->ExtractCellWidget->SetInteractor(this->GetMimxMainWindow()->GetRenderWidget()
                        //              ->GetRenderWindowInteractor());
                        //      this->ExtractCellWidget->SetInput(this->SelectCellsWidget->GetCurrentSelectedSubset());
                        //      this->ExtractCellWidget->SetInputActor(this->BBoxList
                        //              ->GetItem(combobox->GetValueIndex(name))->GetActor());
                        //      this->ExtractCellWidget->SetEnabled(1);
                        //      this->DeleteButtonState = 0;
                        //}
                        //else
                        //{
                        //      if(!this->ExtractCellWidget->GetEnabled())
                        //      {
                        //              this->ExtractCellWidget->SetEnabled(1);
                        //      }
                        //}
                //}
                        if(this->RadioButtonSet->GetWidget(3)->GetSelectedState())
                        {
                                this->DoUndoButtonSet->GetWidget(0)->SetEnabled(0);
                                this->DoUndoButtonSet->GetWidget(1)->SetEnabled(0);
                                this->SelectSubsetRadiobuttonSet->SetEnabled(0);
                        }
                        else
                        {
                                this->DoUndoButtonSet->GetWidget(0)->SetEnabled(1);
                                this->DoUndoButtonSet->GetWidget(1)->SetEnabled(1);
                                this->SelectSubsetRadiobuttonSet->SetEnabled(1);
                        }
        }
        int location = 3;
        for (int i=0; i<6; i++)
        {
                if(location != i)
                {
                        if(this->RadioButtonSet->GetWidget(i)->GetSelectedState())
                        {
                                this->RadioButtonSet->GetWidget(i)->SelectedStateOff();
                                return;
                        }
                }
        }

}
//----------------------------------------------------------------------------
void vtkKWMimxEditBBGroup::EditBBSplitCellCallback(int Mode)
{
        //this->GetApplication()->Script(
        //      "pack forget %s", this->RadiusEntry->GetWidgetName());
        if(!Mode)
        {
                if(this->ExtractEdgeWidget)
                {
                        if(this->ExtractEdgeWidget->GetEnabled())
                        {
                                this->ExtractEdgeWidget->SetEnabled(0);
                        }
                }
                this->EntryFrame->Unpack();
                this->SetDoUndoButtonSelectSubsetButton();
                if(!this->SelectCellsWidget)    return;
                if(!this->SelectCellsWidget->GetCurrentSelectedSubsetActor()) return;
                this->GetMimxMainWindow()->GetRenderWidget()->AddViewProp(
                        this->SelectCellsWidget->GetCurrentSelectedSubsetActor());
                this->GetMimxMainWindow()->GetRenderWidget()->Render();
                return;
        }

        this->CancelStatus = 1;
        this->EntryFrame->Unpack();
        this->CancelStatus = 0;
        //      this->RadiusEntry->SetEnabled(0);
        this->SelectSubsetRadiobuttonSet->SetEnabled(0);
        vtkMimxErrorCallback *callback = this->GetMimxMainWindow()->GetErrorCallback();
        if(!strcmp(this->ObjectListComboBox->GetWidget()->GetValue(),""))
        {
                if(!this->CancelStatus)
                {
                        callback->ErrorMessage("Building Block selection required");
                        this->DeselectAllButtons();
                }
        }
        else{

                vtkKWComboBox *combobox = this->ObjectListComboBox->GetWidget();
                const char *name = combobox->GetValue();
                int num = combobox->GetValueIndex(name);
                if(num < 0 || num > combobox->GetNumberOfValues()-1)
                {
                        callback->ErrorMessage("Choose valid Building-block structure");
                        combobox->SetValue("");
                        this->DeselectAllButtons();
                        return;
                }
                vtkUnstructuredGrid *ugrid = vtkMimxUnstructuredGridActor::SafeDownCast(
                        this->BBoxList->GetItem(combobox->GetValueIndex(name)))->GetDataSet();

                if(!this->SelectCellsWidget)
                {
                        this->SelectCellsWidget = vtkMimxSelectCellsWidget::New();
                        //this->SelectCellsWidget->SetInputActor(this->BBoxList
                        //      ->GetItem(combobox->GetValueIndex(name))->GetActor());
                        this->SelectCellsWidget->SetInput(ugrid);
                        this->SelectCellsWidget->SetInteractor(
                                this->GetMimxMainWindow()->GetRenderWidget()->GetRenderWindowInteractor());
                        //this->SelectCellsWidget->Initialize();
                }

                //if(this->SplitButtonState)
                //{
                  if(this->ExtractEdgeWidget)
                  {
                        if(this->ExtractEdgeWidget->GetEnabled())
                        {
                          this->ExtractEdgeWidget->SetEnabled(0);
                        }
                        this->ExtractEdgeWidget->Delete();
                        this->ExtractEdgeWidget = NULL;
                  }
                  this->ExtractEdgeWidget = vtkMimxExtractEdgeWidget::New();
                  this->ExtractEdgeWidget->SetInteractor(this->GetMimxMainWindow()->GetRenderWidget()
                        ->GetRenderWindowInteractor());
                  this->ExtractEdgeWidget->SetInput(this->SelectCellsWidget->GetCurrentSelectedSubset());
                  //this->ExtractEdgeWidget->SetInputActor(this->BBoxList
                         // ->GetItem(combobox->GetValueIndex(name))->GetActor());
                  this->ExtractEdgeWidget->SetEnabled(
                          this->RadioButtonSet->GetWidget(1)->GetSelectedState());
                  this->SplitButtonState = 0;
                //}
                //else
                //{
                 // if(!this->ExtractEdgeWidget)
                 // {
                        //this->ExtractEdgeWidget = vtkMimxExtractEdgeWidget::New();
                        //this->ExtractEdgeWidget->SetInteractor(this->GetMimxMainWindow()->
                        //      GetRenderWidget()->GetRenderWindowInteractor());
                        //this->ExtractEdgeWidget->SetInput(this->SelectCellsWidget->GetCurrentSelectedSubset());
                        //this->ExtractEdgeWidget->SetInputActor(this->BBoxList
                        //      ->GetItem(combobox->GetValueIndex(name))->GetActor());
                        //this->ExtractEdgeWidget->SetEnabled(1);
                        //this->SplitButtonState = 0;
                 // }
                 // else
                 // {
                        //if(!this->ExtractEdgeWidget->GetEnabled())
                        //{
                        //  this->ExtractEdgeWidget->SetEnabled(1);
                        //}
                 // }
                //}
                if(this->RadioButtonSet->GetWidget(1)->GetSelectedState())
                {
                        this->DoUndoButtonSet->GetWidget(0)->SetEnabled(0);
                        this->DoUndoButtonSet->GetWidget(1)->SetEnabled(0);
                        this->SelectSubsetRadiobuttonSet->SetEnabled(0);
                }
                else
                {
                        this->DoUndoButtonSet->GetWidget(0)->SetEnabled(1);
                        this->DoUndoButtonSet->GetWidget(1)->SetEnabled(1);
                        this->SelectSubsetRadiobuttonSet->SetEnabled(1);
                }
        }
        int location = 1;
        for (int i=0; i<6; i++)
        {
                if(location != i)
                {
                        if(this->RadioButtonSet->GetWidget(i)->GetSelectedState())
                        {
                                this->RadioButtonSet->GetWidget(i)->SelectedStateOff();
                                return;
                        }
                }
        }
}
//----------------------------------------------------------------------------
void vtkKWMimxEditBBGroup::EditBBAddCellCallback(int Mode)
{
        if(!Mode)
        {
                if(this->ExtractFaceWidget)
                {
                        if(this->ExtractFaceWidget->GetEnabled())
                        {
                                this->ExtractFaceWidget->SetEnabled(0);
                        }
                }
                this->EntryFrame->Unpack();
                this->SetDoUndoButtonSelectSubsetButton();
                if(!this->SelectCellsWidget)    return;
                if(!this->SelectCellsWidget->GetCurrentSelectedSubsetActor()) return;
                this->GetMimxMainWindow()->GetRenderWidget()->AddViewProp(
                        this->SelectCellsWidget->GetCurrentSelectedSubsetActor());
                this->GetMimxMainWindow()->GetRenderWidget()->Render();
                return;
        }

        vtkMimxErrorCallback *callback = this->GetMimxMainWindow()->GetErrorCallback();
        if(!strcmp(this->ObjectListComboBox->GetWidget()->GetValue(),""))
        {
                if(!this->CancelStatus)
                {
                        callback->ErrorMessage("Building Block selection required");
                        this->DeselectAllButtons();
                }
        }
        else{
                vtkKWComboBox *combobox = this->ObjectListComboBox->GetWidget();
                const char *name = combobox->GetValue();
                int num = combobox->GetValueIndex(name);
                if(num < 0 || num > combobox->GetNumberOfValues()-1)
                {
                        callback->ErrorMessage("Choose valid Building-block structure");
                        combobox->SetValue("");
                        this->DeselectAllButtons();
                        return;
                }

                vtkUnstructuredGrid *ugrid = vtkMimxUnstructuredGridActor::SafeDownCast(this->BBoxList
                        ->GetItem(combobox->GetValueIndex(name)))->GetDataSet();

                if(!this->SelectCellsWidget)
                {
                        this->SelectCellsWidget = vtkMimxSelectCellsWidget::New();
                        this->SelectCellsWidget->SetInput(ugrid);
                        this->SelectCellsWidget->SetInteractor(
                                this->GetMimxMainWindow()->GetRenderWidget()->GetRenderWindowInteractor());
                        this->SelectCellsWidget->Initialize();
                }

                //if(this->AddButtonState)
                //{
std::cout << "Add Box: " << name << std::endl;
                  if(this->ExtractFaceWidget)
                  {
                        if(this->ExtractFaceWidget->GetEnabled())
                        {
                          this->ExtractFaceWidget->SetEnabled(0);
                        }
                        this->ExtractFaceWidget->Delete();
                        this->ExtractFaceWidget = NULL;
                  }
                  this->ExtractFaceWidget = vtkMimxExtractFaceWidget::New();
                  this->ExtractFaceWidget->SetInteractor(this->GetMimxMainWindow()->GetRenderWidget()
                        ->GetRenderWindowInteractor());
                  this->ExtractFaceWidget->SetCompleteUGrid(ugrid);
                  this->ExtractFaceWidget->SetInput(this->SelectCellsWidget->GetCurrentSelectedSubset());
                  //this->ExtractFaceWidget->SetInputActor(this->BBoxList
                         // ->GetItem(combobox->GetValueIndex(name))->GetActor());
                  //this->ExtractFaceWidget->SetInputActor(this->BBoxList
                         // ->GetItem(combobox->GetValueIndex(name))->GetActor());
                  this->ExtractFaceWidget->SetEnabled(
                          this->RadioButtonSet->GetWidget(2)->GetSelectedState());
                  this->AddButtonState = 0;
                //}
                //else
                //{
                 // if(!this->ExtractFaceWidget)
                 // {
                        //vtkUnstructuredGrid *ugrid = vtkMimxUnstructuredGridActor::SafeDownCast(
                        //  this->BBoxList->GetItem(combobox->GetValueIndex(name)))->GetDataSet();
                        //this->ExtractFaceWidget = vtkMimxExtractFaceWidget::New();
                        //this->ExtractFaceWidget->SetInteractor(this->GetMimxMainWindow()->GetRenderWidget()
                        //  ->GetRenderWindowInteractor());
                        //this->ExtractFaceWidget->SetInput(this->SelectCellsWidget->GetCurrentSelectedSubset());
                        ////this->ExtractFaceWidget->SetInputActor(this->BBoxList
                        ////    ->GetItem(combobox->GetValueIndex(name))->GetActor());
                        //this->ExtractFaceWidget->SetEnabled(1);
                        //this->AddButtonState = 0;
                 // }
                 // else
                 // {
                        //if(!this->ExtractFaceWidget->GetEnabled())
                        //{
                        //  this->ExtractFaceWidget->SetEnabled(1);
                        //}
                 // }
                //}
                  if(this->RadioButtonSet->GetWidget(2)->GetSelectedState())
                  {
                          this->DoUndoButtonSet->GetWidget(0)->SetEnabled(0);
                          this->DoUndoButtonSet->GetWidget(1)->SetEnabled(0);
                          this->SelectSubsetRadiobuttonSet->SetEnabled(0);
                  }
                  else
                  {
                          this->DoUndoButtonSet->GetWidget(0)->SetEnabled(1);
                          this->DoUndoButtonSet->GetWidget(1)->SetEnabled(1);
                          this->SelectSubsetRadiobuttonSet->SetEnabled(1);
                  }
        }
        int location = 2;
        for (int i=0; i<6; i++)
        {
                if(location != i)
                {
                        if(this->RadioButtonSet->GetWidget(i)->GetSelectedState())
                        {
                                this->RadioButtonSet->GetWidget(i)->SelectedStateOff();
                        }
                }
        }

        this->RepackEntryFrame();

        this->RadiusEntry->SetLabelText("Extrusion Length");
        this->RadiusEntry->GetWidget()->SetValueAsDouble( this->defaultExtrusionLength );
        this->RadiusEntry->SetEnabled(1);
}
//----------------------------------------------------------------------------
void vtkKWMimxEditBBGroup::EditBBMoveCellCallback(int Mode)
{
        if(!Mode)
        {
                if(this->UnstructuredGridWidget)
                {
                        if(this->UnstructuredGridWidget->GetEnabled())
                        {
                                this->UnstructuredGridWidget->SetEnabled(0);
                        }
                }
                this->EntryFrame->Unpack();
//              this->RadiusEntry->Unpack();
                this->SetDoUndoButtonSelectSubsetButton();
                if(!this->SelectCellsWidget)    return;
                if(!this->SelectCellsWidget->GetCurrentSelectedSubsetActor()) return;
                this->GetMimxMainWindow()->GetRenderWidget()->AddViewProp(
                        this->SelectCellsWidget->GetCurrentSelectedSubsetActor());
                this->GetMimxMainWindow()->GetRenderWidget()->Render();
                return;
        }

        vtkMimxErrorCallback *callback = this->GetMimxMainWindow()->GetErrorCallback();
        if(!strcmp(this->ObjectListComboBox->GetWidget()->GetValue(),""))
        {
                if(!this->CancelStatus)
                {
                        callback->ErrorMessage("Building Block selection required");
                        this->DeselectAllButtons();
                }
        }
        else
        {
//              this->RadiusEntry->SetEnabled(1);

 /*     if(this->MoveButtonState)
      {*/
        vtkKWComboBox *combobox = this->ObjectListComboBox->GetWidget();
        const char *name = combobox->GetValue();
        int num = combobox->GetValueIndex(name);
                if(num < 0 || num > combobox->GetNumberOfValues()-1)
                {
                        callback->ErrorMessage("Choose valid Building-block structure");
                        combobox->SetValue("");
                        this->DeselectAllButtons();
                        return;
                }
                //
                this->SelectSubsetRadiobuttonSet->SetEnabled(0);
                vtkUnstructuredGrid *ugrid = vtkMimxUnstructuredGridActor::SafeDownCast(this->BBoxList
                        ->GetItem(combobox->GetValueIndex(name)))->GetDataSet();
                if(!this->SelectCellsWidget)
                {
                        this->SelectCellsWidget = vtkMimxSelectCellsWidget::New();
                        this->SelectCellsWidget->SetInput(ugrid);
                        this->SelectCellsWidget->SetInteractor(
                                this->GetMimxMainWindow()->GetRenderWidget()->GetRenderWindowInteractor());
                }
                this->GetMimxMainWindow()->GetRenderWidget()->RemoveViewProp(
                        this->SelectCellsWidget->GetCurrentSelectedSubsetActor());
                //
         if(this->UnstructuredGridWidget)
        {
          if(this->UnstructuredGridWidget->GetEnabled())
          {
            this->UnstructuredGridWidget->SetEnabled(0);
          }
          this->UnstructuredGridWidget->Delete();
          this->UnstructuredGridWidget = NULL;
        }
        this->UnstructuredGridWidget = vtkMimxUnstructuredGridWidget::New();
        this->UnstructuredGridWidget->SetUGrid(this->SelectCellsWidget->GetCurrentSelectedSubset());
                this->UnstructuredGridWidget->SetCompleteUGrid(ugrid);
        this->UnstructuredGridWidget->SetInteractor(this->GetMimxMainWindow()->GetRenderWidget()
          ->GetRenderWindowInteractor());
                if(this->defaultRadiusEntry == -1.0)
                {
                        this->UnstructuredGridWidget->
                                ComputeAverageHandleSize(this->UnstructuredGridWidget);
                        this->Parameter = 1.0/this->UnstructuredGridWidget->GetHandleSize();
                        this->RadiusEntry->GetWidget()->SetValueAsDouble(
                                this->UnstructuredGridWidget->GetHandleSize()*this->Parameter);
                        this->defaultRadiusEntry = this->RadiusEntry->GetWidget()->GetValueAsDouble();
                }
                else
                {
                        this->UnstructuredGridWidget->SetHandleSize(this->RadiusEntry->GetWidget()->
                                GetValueAsDouble()*(1.0/this->Parameter));
                }
                this->UnstructuredGridWidget->Execute(this->UnstructuredGridWidget);
                this->RadiusEntry->SetEnabled(
                        this->RadioButtonSet->GetWidget(0)->GetSelectedState());
        this->MoveButtonState = 0;
      //}
    //  else
    //  {
    //    if(!this->UnstructuredGridWidget)
    //    {
    //      vtkKWComboBox *combobox = this->ObjectListComboBox->GetWidget();
    //      const char *name = combobox->GetValue();
    //      int num = combobox->GetValueIndex(name);
                  //if(num < 0 || num > combobox->GetNumberOfValues()-1)
                  //{
                         // callback->ErrorMessage("Choose valid Building-block structure");
                         // combobox->SetValue("");
                         // return;
                  //}
                  ////
                  //this->SelectSubsetRadiobuttonSet->SetEnabled(0);
                  //if(!this->SelectCellsWidget)
                  //{
                         // this->SelectCellsWidget = vtkMimxSelectCellsWidget::New();
                         // this->SelectCellsWidget->SetInputActor(this->BBoxList
                                //  ->GetItem(combobox->GetValueIndex(name))->GetActor());
                         // this->SelectCellsWidget->SetInteractor(
                                //  this->GetMimxMainWindow()->GetRenderWidget()->GetRenderWindowInteractor());
                         // this->SelectCellsWidget->Initialize();
                  //}
                  ////
    //      //vtkUnstructuredGrid *ugrid = vtkMimxUnstructuredGridActor::SafeDownCast(this->BBoxList
    //      //  ->GetItem(combobox->GetValueIndex(name)))->GetDataSet();
    //      this->UnstructuredGridWidget = vtkMimxUnstructuredGridWidget::New();
    //      this->UnstructuredGridWidget->SetUGrid(this->SelectCellsWidget->GetCurrentSelectedSubset());
    //      this->UnstructuredGridWidget->SetInteractor(this->GetMimxMainWindow()->
    //        GetRenderWidget()->GetRenderWindowInteractor());
    //      this->UnstructuredGridWidget->Execute();
                  //this->UnstructuredGridWidget->SetHandleSize(
                         // this->RadiusEntry->GetWidget()->GetValueAsDouble());
    //      this->UnstructuredGridWidget->SetEnabled(1);
    //      this->MoveButtonState = 0;
                  //this->RadiusEntry->SetEnabled(1);
    //    }
    //    else
    //    {
    //      if(!this->UnstructuredGridWidget->GetEnabled())
    //      {
                         // this->UnstructuredGridWidget->SetHandleSize(
                                //  this->RadiusEntry->GetWidget()->GetValueAsDouble());
    //        this->UnstructuredGridWidget->SetEnabled(1);
    //      }
    //    }
    //  }

                if(this->RadioButtonSet->GetWidget(0)->GetSelectedState())
                {
                        this->DoUndoButtonSet->GetWidget(0)->SetEnabled(0);
                        this->DoUndoButtonSet->GetWidget(1)->SetEnabled(0);
                        this->SelectSubsetRadiobuttonSet->SetEnabled(0);
                }
                else
                {
                        this->DoUndoButtonSet->GetWidget(0)->SetEnabled(1);
                        this->DoUndoButtonSet->GetWidget(1)->SetEnabled(1);
                        this->SelectSubsetRadiobuttonSet->SetEnabled(1);
                }
        }

        int location = 0;
        for (int i=0; i<6; i++)
        {
                if(location != i)
                {
                        if(this->RadioButtonSet->GetWidget(i)->GetSelectedState())
                        {
                                this->RadioButtonSet->GetWidget(i)->SelectedStateOff();
                        }
                }
        }
        this->RepackEntryFrame();

        this->RadiusEntry->SetLabelText("Radius");
        this->RadiusEntry->GetWidget()->SetValueAsDouble(this->defaultRadiusEntry);
}

void vtkKWMimxEditBBGroup::EditBBVtkInteractionCallback(int Mode)
{
  if(this->UnstructuredGridWidget)
  {
    if (this->UnstructuredGridWidget->GetEnabled())
    {
      this->UnstructuredGridWidget->SetEnabled(0);
          
    }
        
  }
  if(this->ExtractEdgeWidget)
  {
    if(this->ExtractEdgeWidget->GetEnabled())
    {
      this->ExtractEdgeWidget->SetEnabled(0);
    }
  }
  if(this->ExtractFaceWidget)
  {
          if(this->ExtractFaceWidget->GetEnabled())
          {
                  this->ExtractFaceWidget->SetEnabled(0);
          }
  }
  if(this->ExtractCellWidget)
  {
          if(this->ExtractCellWidget->GetEnabled())
          {
                  this->ExtractCellWidget->SetEnabled(0);
                  return;
          }
  }
   if(this->ExtractCellWidgetHBB)
  {
          if(this->ExtractCellWidgetHBB->GetEnabled())
          {
                  this->ExtractCellWidgetHBB->SetEnabled(0);
                  return;
          }
  }
  
   if(this->SelectCellsWidget)
   {
           if(this->SelectCellsWidget->GetCurrentSelectedSubsetActor())
           {
                   this->GetMimxMainWindow()->GetRenderWidget()->RemoveViewProp(
                           this->SelectCellsWidget->GetCurrentSelectedSubsetActor());
           }
           if(this->SelectCellsWidget->GetCurrentSelectedSubsetActor())
           {
                   this->GetMimxMainWindow()->GetRenderWidget()->RemoveViewProp(
                           this->SelectCellsWidget->GetCurrentSelectedSubsetActor());
           }
           this->SelectCellsWidget->SetEnabled(0);
           //this->SelectCellsWidget->Delete();
           //this->SelectCellsWidget = NULL;
           this->GetMimxMainWindow()->GetRenderWidget()->Render();
   }

   vtkKWComboBox *combobox = this->ObjectListComboBox->GetWidget();
  if(strcmp(combobox->GetValue(),""))
  {
          const char *name = combobox->GetValue();
          int num = combobox->GetValueIndex(name);
std::cout << "Item Name: " << name << std::endl;
std::cout << "Item Index: " << num << std::endl;
std::cout << "Item: " << this->DoUndoTree->GetItem(name) << std::endl;
std::cout << "Item 0: " << this->DoUndoTree->GetItem(static_cast<vtkIdType>(0)) << std::endl;
std::cout << "Item 1: " << this->DoUndoTree->GetItem(static_cast<vtkIdType>(1)) << std::endl;
std::cout << "Call Get GetItemNumber " << this->DoUndoTree->GetItemNumber(name) << std::endl;
std::cout << "Item Child: " << this->DoUndoTree->GetItem(name)->Child << std::endl;

          if(this->DoUndoTree->GetItem(name)->Child != NULL)
          {
                  this->DoUndoButtonSet->GetWidget(1)->SetEnabled(1);
          }
          else{
                  this->DoUndoButtonSet->GetWidget(1)->SetEnabled(0);
          }
          if(this->DoUndoTree->GetItem(name)->Parent != NULL)
          {
                  this->DoUndoButtonSet->GetWidget(0)->SetEnabled(1);
          }
          else
          {
                  this->DoUndoButtonSet->GetWidget(0)->SetEnabled(0);
          }
  }
  else
  {
            this->DoUndoButtonSet->GetWidget(1)->SetEnabled(0);
                this->DoUndoButtonSet->GetWidget(0)->SetEnabled(0);
  }
  this->CancelStatus = 1;
  this->EntryFrame->Unpack();
  this->CancelStatus = 0;
  //    this->RadiusEntry->SetEnabled(0);
  //this->SelectSubsetRadiobuttonSet->SetEnabled(1);
}
//-----------------------------------------------------------------------------
void vtkKWMimxEditBBGroup::UpdateObjectLists()
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
        this->SelectSubsetRadiobuttonSet->SetEnabled(1);
  }
        else
        {
                this->SelectSubsetRadiobuttonSet->SetEnabled(0);
        }
  this->SetDoUndoButtonSelectSubsetButton(); 
  return;
}
//--------------------------------------------------------------------------------
void vtkKWMimxEditBBGroup::RadiusChangeCallback(const char *dummy)
{
        if(!this->CancelStatus)
        {
                if(this->UnstructuredGridWidget)
                {
                        if(this->UnstructuredGridWidget->GetEnabled())
                        {
                                this->UnstructuredGridWidget->SetEnabled(0);
                                this->UnstructuredGridWidget->SetHandleSize(this->RadiusEntry->
                                        GetWidget()->GetValueAsDouble()*(1.0/this->Parameter));
                                this->defaultRadiusEntry = this->Parameter*
                                        this->RadiusEntry->GetWidget()->GetValueAsDouble();
                                this->UnstructuredGridWidget->SetEnabled(1);
                                this->GetMimxMainWindow()->GetRenderWidget()->Render();
        //                      this->GetMimxMainWindow()->GetRenderWidget()->ResetCamera();
                        }
                }
        }
}
//--------------------------------------------------------------------------------
void vtkKWMimxEditBBGroup::EditBBDoneCallback()
{
        if(this->EditBBApplyCallback())
                this->EditBBCancelCallback();
}
//---------------------------------------------------------------------------------
void vtkKWMimxEditBBGroup::EditBBMirrorCallback(int Mode)
{
        if(!Mode)
        {
                this->MirrorFrame->Unpack();
                this->SetDoUndoButtonSelectSubsetButton();
                //
                if(this->AxisSelection->GetWidget(0)->GetSelectedState())
                {
                        this->PlaceMirroringPlaneAboutX();
                        return;
                }
                if(this->AxisSelection->GetWidget(1)->GetSelectedState())
                {
                        this->PlaceMirroringPlaneAboutY();
                        return;
                }
                if(this->AxisSelection->GetWidget(2)->GetSelectedState())
                {
                        this->PlaceMirroringPlaneAboutZ();
                        return;
                }
                //
                if(this->MirrorPlaneWidget)
                {this->MirrorPlaneWidget->SetEnabled(0);}
                if(!this->SelectCellsWidget)    return;
                if(!this->SelectCellsWidget->GetCurrentSelectedSubsetActor()) return;
                this->GetMimxMainWindow()->GetRenderWidget()->AddViewProp(
                        this->SelectCellsWidget->GetCurrentSelectedSubsetActor());
                this->GetMimxMainWindow()->GetRenderWidget()->Render();
                return;
        }

        vtkMimxErrorCallback *callback = this->GetMimxMainWindow()->GetErrorCallback();
        if(!strcmp(this->ObjectListComboBox->GetWidget()->GetValue(),""))
        {
                if(!this->CancelStatus)
                {
                        callback->ErrorMessage("Building Block selection required");
                        this->DeselectAllButtons();
                }
                return;
        }
        else
        {
                vtkKWComboBox *combobox = this->ObjectListComboBox->GetWidget();
                const char *name = combobox->GetValue();
                vtkUnstructuredGrid *ugrid = vtkMimxUnstructuredGridActor::SafeDownCast(
                        this->BBoxList->GetItem(combobox->GetValueIndex(name)))->GetDataSet();
                if(!this->SelectCellsWidget)
                {
                        this->SelectCellsWidget = vtkMimxSelectCellsWidget::New();
                        this->SelectCellsWidget->SetInteractor(
                                this->GetMimxMainWindow()->GetRenderWidget()->GetRenderWindowInteractor());
                }
                this->SelectCellsWidget->SetInput(ugrid);
                //
                if(this->RadioButtonSet->GetWidget(0)->GetSelectedState())
                {
                        this->DoUndoButtonSet->GetWidget(0)->SetEnabled(0);
                        this->DoUndoButtonSet->GetWidget(1)->SetEnabled(0);
                        this->SelectSubsetRadiobuttonSet->SetEnabled(0);
                }
                else
                {
                        this->DoUndoButtonSet->GetWidget(0)->SetEnabled(1);
                        this->DoUndoButtonSet->GetWidget(1)->SetEnabled(1);
                        this->SelectSubsetRadiobuttonSet->SetEnabled(1);
                }
        }

        int location = 4;
        for (int i=0; i<6; i++)
        {
                if(location != i)
                {
                        if(this->RadioButtonSet->GetWidget(i)->GetSelectedState())
                        {
                                this->RadioButtonSet->GetWidget(i)->SelectedStateOff();
                        }
                }
        }
        this->RepackMirrorFrame();
        this->GetMimxMainWindow()->GetRenderWidget()->Render();
}
//----------------------------------------------------------------------------------
void vtkKWMimxEditBBGroup::EditBBMergeCallback(int Mode)
{
        //if (!this->MergeBBGroup)
        //{
        //      this->MergeBBGroup = vtkKWMimxMergeBBGroup::New();
        //      this->MergeBBGroup->SetApplication(this->GetApplication());
        //      this->MergeBBGroup->SetParent(this->GetParent());
        //      this->MergeBBGroup->SetSurfaceList(this->SurfaceList);
        //      this->MergeBBGroup->SetBBoxList(this->BBoxList);
        //      this->MergeBBGroup->SetMimxMainWindow(this->GetMimxMainWindow());
        //      this->MergeBBGroup->SetViewProperties(this->ViewProperties);
        //      this->MergeBBGroup->SetMenuGroup(this->MenuGroup);
        //      this->MergeBBGroup->SetEditBBGroup(this);
        //      this->MergeBBGroup->SetVTKRadioButton(//this->RadioButtonSet->GetWidget(6));
        //      this->MergeBBGroup->SetDoUndoTree(this->DoUndoTree);
        //      this->MergeBBGroup->Create();
        //}
        //else
        //{
        //      this->MergeBBGroup->UpdateObjectLists();
        //}
        //this->EditBBCancelCallback();
        //this->MenuGroup->SetMenuButtonsEnabled(0);

        //this->GetApplication()->Script(
        //      "pack %s -side bottom -anchor nw -expand y -padx 0 -pady 2", 
        //      this->MergeBBGroup->GetMainFrame()->GetWidgetName()); 

        if(!Mode)
        {
                this->EntryFrame->Unpack();
                this->SetDoUndoButtonSelectSubsetButton();
                if(!this->SelectCellsWidget)    return;
                if(!this->SelectCellsWidget->GetCurrentSelectedSubsetActor()) return;
                this->GetMimxMainWindow()->GetRenderWidget()->AddViewProp(
                        this->SelectCellsWidget->GetCurrentSelectedSubsetActor());
                this->GetMimxMainWindow()->GetRenderWidget()->Render();
                return;
        }
        vtkMimxErrorCallback *callback = this->GetMimxMainWindow()->GetErrorCallback();
        if(!strcmp(this->ObjectListComboBox->GetWidget()->GetValue(),""))
        {
                if(!this->CancelStatus)
                {
                        callback->ErrorMessage("Building Block selection required");
                        this->DeselectAllButtons();
                        return;
                }
        }
        else{

                vtkKWComboBox *combobox = this->ObjectListComboBox->GetWidget();
                const char *name = combobox->GetValue();
                int num = combobox->GetValueIndex(name);
                if(num < 0 || num > combobox->GetNumberOfValues()-1)
                {
                        callback->ErrorMessage("Choose valid Building-block structure");
                        combobox->SetValue("");
                        this->DeselectAllButtons();
                        return;
                }
                vtkUnstructuredGrid *ugrid = vtkMimxUnstructuredGridActor::SafeDownCast(
                        this->BBoxList->GetItem(combobox->GetValueIndex(name)))->GetDataSet();

                if(!this->SelectCellsWidget)
                {
                        this->SelectCellsWidget = vtkMimxSelectCellsWidget::New();
                        this->SelectCellsWidget->SetInput(ugrid);
                        this->SelectCellsWidget->SetInteractor(
                                this->GetMimxMainWindow()->GetRenderWidget()->GetRenderWindowInteractor());
                        this->SelectCellsWidget->Initialize();
                }

        this->SelectSubsetRadiobuttonSet->SetEnabled(0);

        }
        
        int location = 5;
        for (int i=0; i<6; i++)
        {
                if(location != i)
                {
                        if(this->RadioButtonSet->GetWidget(i)->GetSelectedState())
                        {
                                this->RadioButtonSet->GetWidget(i)->SelectedStateOff();
                        }
                }
        }
        this->RepackEntryFrame();

        this->RadiusEntry->SetLabelText("Merge Tolerance");
        this->RadiusEntry->GetWidget()->SetValueAsDouble( defaultMergeTolerance );
        this->RadiusEntry->SetEnabled(1);
}
//------------------------------------------------------------------------------------
void vtkKWMimxEditBBGroup::EditConvertToHBBCallback(int Mode)
{
        this->CancelStatus = 1;
        this->EntryFrame->Unpack();
        this->CancelStatus = 0;
        //      this->RadiusEntry->SetEnabled(0);
        vtkMimxErrorCallback *callback = this->GetMimxMainWindow()->GetErrorCallback();
        if(!strcmp(this->ObjectListComboBox->GetWidget()->GetValue(),""))
        {
                if(!this->CancelStatus)
                {
                        callback->ErrorMessage("Building Block selection required");
                        this->DeselectAllButtons();
                }
        }
        else
        {
                if(this->ConvertToHBBButtonState)
                {
                        vtkKWComboBox *combobox = this->ObjectListComboBox->GetWidget();
                        const char *name = combobox->GetValue();
                        int num = combobox->GetValueIndex(name);
                        if(num < 0 || num > combobox->GetNumberOfValues()-1)
                        {
                                callback->ErrorMessage("Choose valid Building-block structure");
                                combobox->SetValue("");
                                this->DeselectAllButtons();
                                return;
                        }
                        vtkUnstructuredGrid *ugrid = vtkMimxUnstructuredGridActor::SafeDownCast(this->BBoxList
                                ->GetItem(combobox->GetValueIndex(name)))->GetDataSet();
                        if(this->ExtractCellWidgetHBB)
                        {
                                if(this->ExtractCellWidgetHBB->GetEnabled())
                                {
                                        this->ExtractCellWidgetHBB->SetEnabled(0);
                                }
                                this->ExtractCellWidgetHBB->Delete();
                                this->ExtractCellWidgetHBB = NULL;
                        }
                        this->ExtractCellWidgetHBB = vtkMimxExtractCellWidget::New();
                        this->ExtractCellWidgetHBB->SetInteractor(this->GetMimxMainWindow()->GetRenderWidget()
                                ->GetRenderWindowInteractor());
                        this->ExtractCellWidgetHBB->SetInput(ugrid);
                        this->ExtractCellWidgetHBB->SetInputActor(this->BBoxList
                                ->GetItem(combobox->GetValueIndex(name))->GetActor());
                        this->ExtractCellWidgetHBB->SetEnabled(1);
                        this->ConvertToHBBButtonState = 0;
                }
                else
                {
                        if(!this->ExtractCellWidgetHBB)
                        {
                                vtkKWComboBox *combobox = this->ObjectListComboBox->GetWidget();
                                const char *name = combobox->GetValue();
                                int num = combobox->GetValueIndex(name);
                                if(num < 0 || num > combobox->GetNumberOfValues()-1)
                                {
                                        callback->ErrorMessage("Choose valid Building-block structure");
                                        combobox->SetValue("");
                                        this->DeselectAllButtons();
                                        return;
                                }
                                vtkUnstructuredGrid *ugrid = vtkMimxUnstructuredGridActor::SafeDownCast(
                                        this->BBoxList->GetItem(combobox->GetValueIndex(name)))->GetDataSet();
                                this->ExtractCellWidgetHBB = vtkMimxExtractCellWidget::New();
                                this->ExtractCellWidgetHBB->SetInteractor(this->GetMimxMainWindow()->GetRenderWidget()
                                        ->GetRenderWindowInteractor());
                                this->ExtractCellWidgetHBB->SetInput(ugrid);
                                this->ExtractCellWidgetHBB->SetInputActor(this->BBoxList
                                        ->GetItem(combobox->GetValueIndex(name))->GetActor());
                                this->ExtractCellWidgetHBB->SetEnabled(1);
                                this->ConvertToHBBButtonState = 0;
                        }
                        else
                        {
                                if(!this->ExtractCellWidgetHBB->GetEnabled())
                                {
                                        this->ExtractCellWidgetHBB->SetEnabled(1);
                                }
                        }
                }
                this->DoUndoButtonSet->GetWidget(0)->SetEnabled(0);
                this->DoUndoButtonSet->GetWidget(1)->SetEnabled(0);
        }
        if(UnstructuredGridWidget)
        {
                if(UnstructuredGridWidget->GetEnabled())
                {
                        UnstructuredGridWidget->SetEnabled(0);
                        return;
                }
        }
        if(this->ExtractEdgeWidget)
        {
                if(this->ExtractEdgeWidget->GetEnabled())
                {
                        this->ExtractEdgeWidget->SetEnabled(0);
                        return;
                }
        }
        if(this->ExtractFaceWidget)
        {
                if(this->ExtractFaceWidget->GetEnabled())
                {
                        this->ExtractFaceWidget->SetEnabled(0);
                        return;
                }
        }

        if(this->ExtractCellWidget)
        {
                if(this->ExtractCellWidget->GetEnabled())
                {
                        this->ExtractCellWidget->SetEnabled(0);
                        return;
                }
        }
}
//----------------------------------------------------------------------------------------------------
void vtkKWMimxEditBBGroup::DoBBCallback()
{
        vtkKWComboBox *combobox = this->ObjectListComboBox->GetWidget();
        if(strcmp(combobox->GetValue(),""))
        {
                const char *name = combobox->GetValue();
                int num = combobox->GetValueIndex(name);
                if(this->DoUndoTree->GetItem(name)->Child != NULL)
                {
                        Node *childnode = this->DoUndoTree->GetItem(name)->Child;
                        combobox->DeleteValue(num);
                        combobox->AddValue(childnode->Data->GetFileName());
                        combobox->SetValue(childnode->Data->GetFileName());
                        this->GetMimxMainWindow()->GetRenderWidget()->RemoveViewProp(
                                this->BBoxList->GetItem(num)->GetActor());
                        this->BBoxList->RemoveItem(num);
                        this->BBoxList->AppendItem(childnode->Data);
                        this->GetMimxMainWindow()->GetRenderWidget()->AddViewProp(
                                this->BBoxList->GetItem(this->BBoxList->GetNumberOfItems()-1)->GetActor());
                        this->GetMimxMainWindow()->GetViewProperties()->DeleteObjectList(5, num);
                        this->GetMimxMainWindow()->GetViewProperties()->AddObjectList(
                                this->BBoxList->GetItem(this->BBoxList->GetNumberOfItems()-1));
                        this->SelectionChangedCallback(combobox->GetValue());
                }
        }
}
//----------------------------------------------------------------------------------------------------
void vtkKWMimxEditBBGroup::UndoBBCallback()
{
        vtkKWComboBox *combobox = this->ObjectListComboBox->GetWidget();
        if(strcmp(combobox->GetValue(),""))
        {
                const char *name = combobox->GetValue();
                int num = combobox->GetValueIndex(name);
                if(this->DoUndoTree->GetItem(name)->Parent != NULL)
                {
                        Node *parentnode = this->DoUndoTree->GetItem(name)->Parent;
                        combobox->DeleteValue(num);
                        combobox->AddValue(parentnode->Data->GetFileName());
                        combobox->SetValue(parentnode->Data->GetFileName());
                        this->GetMimxMainWindow()->GetRenderWidget()->RemoveViewProp(
                                this->BBoxList->GetItem(num)->GetActor());
                        this->BBoxList->RemoveItem(num);
                        this->BBoxList->AppendItem(parentnode->Data);
                        this->GetMimxMainWindow()->GetRenderWidget()->AddViewProp(
                                this->BBoxList->GetItem(this->BBoxList->GetNumberOfItems()-1)->GetActor());
                        this->GetMimxMainWindow()->GetViewProperties()->DeleteObjectList(5, num);
                        this->GetMimxMainWindow()->GetViewProperties()->AddObjectList(
                                this->BBoxList->GetItem(this->BBoxList->GetNumberOfItems()-1));
                        this->SelectionChangedCallback(combobox->GetValue());
                }
        }
}
//----------------------------------------------------------------------------------------------------
void vtkKWMimxEditBBGroup::AddEditedBB(int BBNum, vtkUnstructuredGrid *Output, 
                                                                           const char* Name, vtkIdType& Count )
{
        ////// new code
        // create a new node and store the latest building block structure
        Node *chosennode = this->DoUndoTree->GetItem(
                this->BBoxList->GetItem(BBNum)->GetFileName());
        // enable/disable do/undo buttons
        char *chosennodename = chosennode->Data->GetFileName();
        this->DoUndoButtonSet->GetWidget(0)->SetEnabled(1);
        this->DoUndoButtonSet->GetWidget(1)->SetEnabled(0);
        //
        this->GetMimxMainWindow()->GetRenderWidget()->RemoveViewProp(
                this->BBoxList->GetItem(BBNum)->GetActor());


        // delete all the building-blocks after the current one
        while (chosennode->Child != NULL)
        {
                chosennode = chosennode->Child;
        }

        while (strcmp(chosennodename, chosennode->Data->GetFileName()))
        {
                Node *currnode = chosennode;
                chosennode = chosennode->Parent;
                this->DoUndoTree->RemoveItem(
                        this->DoUndoTree->GetItemNumber(currnode->Data->GetFileName()));
                currnode->Data->Delete();
                delete currnode;
                chosennode->Child = NULL;
        }
        this->DoUndoTree->AppendItem(new Node);
        this->BBoxList->AppendItem(vtkMimxUnstructuredGridActor::New());
        this->DoUndoTree->GetItem(this->DoUndoTree->GetNumberOfItems()-1)
                ->Data = vtkMimxUnstructuredGridActor::SafeDownCast(
                this->BBoxList->GetItem(this->BBoxList->GetNumberOfItems()-1));
        this->DoUndoTree->GetItem(this->DoUndoTree->GetNumberOfItems()-1)
                ->Parent = chosennode;
        chosennode->Child = this->DoUndoTree->GetItem(this->DoUndoTree->GetNumberOfItems()-1);
        this->DoUndoTree->GetItem(this->DoUndoTree->GetNumberOfItems()-1)
                ->Child = NULL;
        this->BBoxList->GetItem(this->BBoxList->GetNumberOfItems()-1)->
                SetDataType(ACTOR_BUILDING_BLOCK);
        ////
        vtkMimxUnstructuredGridActor::SafeDownCast(this->BBoxList->GetItem(
                this->BBoxList->GetNumberOfItems()-1))->GetDataSet()->
                DeepCopy(Output);
        Count++;
        vtkMimxUnstructuredGridActor::SafeDownCast(this->BBoxList->GetItem(
                this->BBoxList->GetNumberOfItems()-1))->SetObjectName(Name,Count);
        vtkMimxUnstructuredGridActor::SafeDownCast(this->BBoxList->GetItem(
                this->BBoxList->GetNumberOfItems()-1))->GetDataSet()->Modified();
        //
        this->BBoxList->GetItem(this->BBoxList->GetNumberOfItems()-1)->GetActor()->
                GetProperty()->SetRepresentation(this->BBoxList->GetItem(BBNum)->GetActor()
                ->GetProperty()->GetRepresentation());
        //
        this->GetMimxMainWindow()->GetRenderWidget()->AddViewProp(
                this->BBoxList->GetItem(this->BBoxList->GetNumberOfItems()-1)->GetActor());
        this->ObjectListComboBox->GetWidget()->SetValue(
                this->BBoxList->GetItem(this->BBoxList->GetNumberOfItems()-1)->GetFileName());
        this->ObjectListComboBox->GetWidget()->AddValue(
                this->BBoxList->GetItem(this->BBoxList->GetNumberOfItems()-1)->GetFileName());
        this->BBoxList->GetItem(this->BBoxList->GetNumberOfItems()-1)->GetActor()->GetProperty()
                ->SetColor(this->BBoxList->GetItem(BBNum)->GetActor()->GetProperty()->GetColor());
        this->GetMimxMainWindow()->GetRenderWidget()->Render();
        //        this->GetMimxMainWindow()->GetRenderWidget()->ResetCamera();
        this->GetMimxMainWindow()->GetViewProperties()->AddObjectList(
                this->BBoxList->GetItem(this->BBoxList->GetNumberOfItems()-1));
        // remove the parent from the all the lists
        this->BBoxList->RemoveItem(BBNum);
        this->ObjectListComboBox->GetWidget()->DeleteValue(BBNum);
        this->GetMimxMainWindow()->GetViewProperties()->DeleteObjectList(5, BBNum);
        //
        //this->RadioButtonSet->GetWidget(6)->SelectedStateOn();
        this->SelectionChangedCallback(NULL);
        
        /* Update the Mesh Seed to force progation to New Cell */
        /* Should be handled in the Unstructured Grid Filters
        vtkMimxUnstructuredGridActor *ugridActor = vtkMimxUnstructuredGridActor::SafeDownCast(
                            this->BBoxList->GetItem(this->BBoxList->GetNumberOfItems()-1));
        int xSeed = ugridActor->GetMeshSeed(0, 0);
        int ySeed = ugridActor->GetMeshSeed(0, 1);
        int zSeed = ugridActor->GetMeshSeed(0, 2);
        ugridActor->ChangeMeshSeed(0, 0, xSeed);        
        ugridActor->ChangeMeshSeed(0, 1, ySeed);        
        ugridActor->ChangeMeshSeed(0, 2, zSeed);         
        */    
        //
}
//---------------------------------------------------------------------------------------
void vtkKWMimxEditBBGroup::ApplySelectSubsetCallback()
{
//      vtkMimxErrorCallback *callback = this->GetMimxMainWindow()->GetErrorCallback();
//      if(this->SelectCellsWidget)
//      {
//              if (this->SelectCellsWidget->GetEnabled())
//              {
////                    this->SelectCellsWidget->AcceptSelectedMesh();
//                      this->SelectCellsWidget->SetEnabled(0);
//                      this->GetMimxMainWindow()->GetRenderWidget()->GetRenderer()->AddActor(
//                              this->SelectCellsWidget->GetCurrentSelectedSubsetActor());
//                      this->GetMimxMainWindow()->GetRenderWidget()->Render();
//              }
//              else
//              {
//                      callback->ErrorMessage("Enable cells selections");
//              }
//      }
}
//---------------------------------------------------------------------------------------
void vtkKWMimxEditBBGroup::SelectSubsetCallback()
{
        vtkMimxErrorCallback *callback = this->GetMimxMainWindow()->GetErrorCallback();
        if(!strcmp(this->ObjectListComboBox->GetWidget()->GetValue(),""))
        {
                if(!this->CancelStatus)
                {
                        callback->ErrorMessage("Building Block selection required");
                }
        }
        else{
                        vtkKWComboBox *combobox = this->ObjectListComboBox->GetWidget();
                        const char *name = combobox->GetValue();
                        int num = combobox->GetValueIndex(name);
                        if(num < 0 || num > combobox->GetNumberOfValues()-1)
                        {
                                callback->ErrorMessage("Choose valid Building-block structure");
                                combobox->SetValue("");
                                this->DeselectAllButtons();
                                return;
                        }
                        vtkActor *actor = this->BBoxList->GetItem(combobox->GetValueIndex(name))->GetActor();
                        vtkUnstructuredGrid *ugrid = vtkMimxUnstructuredGridActor::SafeDownCast(
                                this->BBoxList->GetItem(combobox->GetValueIndex(name)))->GetDataSet();

                        if(!this->SelectCellsWidget)
                        {
                                this->SelectCellsWidget = vtkMimxSelectCellsWidget::New();
                        }
                        this->SelectCellsWidget->SetInput(ugrid);
                        this->SelectCellsWidget->SetInteractor(
                                this->GetMimxMainWindow()->GetRenderWidget()->GetRenderWindowInteractor());
                        this->GetMimxMainWindow()->GetRenderWidget()->GetRenderer()->RemoveActor(actor);
                        this->SelectCellsWidget->SetEnabled(1);
        }

}
//---------------------------------------------------------------------------------------
void vtkKWMimxEditBBGroup::CancelSelectSubsetCallback()
{
        //if(this->SelectCellsWidget)
        //{
        //      if (this->SelectCellsWidget->GetEnabled())
        //      {
        //              this->SelectCellsWidget->SetEnabled(0);
        //              this->GetMimxMainWindow()->GetRenderWidget()->GetRenderer()->AddActor(
        //                      this->SelectCellsWidget->GetCurrentSelectedSubsetActor());
        //              this->GetMimxMainWindow()->GetRenderWidget()->Render();
        //      }
        //}
}
//---------------------------------------------------------------------------------------
void vtkKWMimxEditBBGroup::SelectFullSetCallback()
{
        vtkMimxErrorCallback *callback = this->GetMimxMainWindow()->GetErrorCallback();
        if(!strcmp(this->ObjectListComboBox->GetWidget()->GetValue(),""))
        {
                if(!this->CancelStatus)
                {
                        callback->ErrorMessage("Building Block selection required");
                }
        }
        else{

                vtkKWComboBox *combobox = this->ObjectListComboBox->GetWidget();
                const char *name = combobox->GetValue();
                int num = combobox->GetValueIndex(name);
                if(num < 0 || num > combobox->GetNumberOfValues()-1)
                {
                        callback->ErrorMessage("Choose valid Building-block structure");
                        combobox->SetValue("");
                        return;
                }
                vtkActor *actor = this->BBoxList->GetItem(combobox->GetValueIndex(name))->GetActor();
                vtkUnstructuredGrid *ugrid = vtkMimxUnstructuredGridActor::SafeDownCast(
                        this->BBoxList->GetItem(combobox->GetValueIndex(name)))->GetDataSet();
                if(this->SelectCellsWidget)
                {
                        //if (this->SelectCellsWidget->GetEnabled())
                        //{
                        //if(this->SelectCellsWidget->GetCurrentSelectedSubsetActor())
                        //      this->GetMimxMainWindow()->GetRenderWidget()->GetRenderer()->RemoveActor(
                        //              this->SelectCellsWidget->GetCurrentSelectedSubsetActor());
                        //      this->SelectCellsWidget->SetInputActor(actor);
                        //      this->SelectCellsWidget->Initialize();
                                this->SelectCellsWidget->SetInput(ugrid);
                                this->SelectCellsWidget->SetEnabled(0);
                                this->GetMimxMainWindow()->GetRenderWidget()->GetRenderer()->RemoveActor(
                                        this->SelectCellsWidget->GetCurrentSelectedSubsetActor());
                        //}
                        //this->SelectCellsWidget->Delete();
                        //this->SelectCellsWidget = NULL;
                }
//              this->SelectCellsWidget = vtkMimxSelectCellsWidget::New();
//              this->SelectCellsWidget->SetInputActor(actor);
//              this->SelectCellsWidget->SetInteractor(
//                      this->GetMimxMainWindow()->GetRenderWidget()->GetRenderWindowInteractor());
//              this->SelectCellsWidget->Initialize();
////            this->SelectCellsWidget->SetEnabled(1);

                this->GetMimxMainWindow()->GetRenderWidget()->GetRenderer()->AddActor(actor);
                this->GetMimxMainWindow()->GetRenderWidget()->Render();
        }       
}
//---------------------------------------------------------------------------------------
void vtkKWMimxEditBBGroup::DeselectAllButtons()
{
        int i;
        for (i=0; i<6; i++)
        {
                if(this->RadioButtonSet->GetWidget(i)->GetSelectedState())
                        this->RadioButtonSet->GetWidget(i)->SelectedStateOff();
        }
        this->EntryFrame->Unpack();
}
//----------------------------------------------------------------------------------------
void vtkKWMimxEditBBGroup::SetDoUndoButtonSelectSubsetButton()
{
        if(!strcmp(this->ObjectListComboBox->GetWidget()->GetValue(),""))
        {
                this->DoUndoButtonSet->SetEnabled(0);
                this->SelectSubsetRadiobuttonSet->SetEnabled(0);
                return;
        }
        vtkKWComboBox *combobox = this->ObjectListComboBox->GetWidget();
        const char *name = combobox->GetValue();
        this->SelectSubsetRadiobuttonSet->SetEnabled(1);
                                
        if(this->DoUndoTree->GetItem(name)->Parent != NULL)     
                this->DoUndoButtonSet->GetWidget(0)->SetEnabled(1);
        else
                this->DoUndoButtonSet->GetWidget(0)->SetEnabled(0);

        if(this->DoUndoTree->GetItem(name)->Child != NULL)      
                this->DoUndoButtonSet->GetWidget(1)->SetEnabled(1);
        else
                this->DoUndoButtonSet->GetWidget(1)->SetEnabled(0);
                
}
//----------------------------------------------------------------------------------------
void vtkKWMimxEditBBGroup::RepackEntryFrame()
{
        this->GetApplication()->Script(
                "pack %s -side top -anchor nw -expand n -padx 2 -pady 2 -after %s", 
                this->EntryFrame->GetWidgetName(), this->RadioButtonSet->GetWidgetName());

        this->GetApplication()->Script(
                "pack %s -side top -anchor nw -expand 0 -padx 2 -pady 6 -fill x", 
                this->RadiusEntry->GetWidgetName());
}
//-----------------------------------------------------------------------------------------
void vtkKWMimxEditBBGroup::PlaceMirroringPlaneAboutX()
{
        vtkMimxErrorCallback *callback = this->GetMimxMainWindow()->GetErrorCallback();
        if(!strcmp(this->ObjectListComboBox->GetWidget()->GetValue(),""))
        {
                callback->ErrorMessage("Building Block selection required");
        }
        else
        {
                vtkKWComboBox *combobox = this->ObjectListComboBox->GetWidget();
                const char *name = combobox->GetValue();
                int num = combobox->GetValueIndex(name);
                if(num < 0 || num > combobox->GetNumberOfValues()-1)
                {
                        callback->ErrorMessage("Choose valid Building-block structure");
                        combobox->SetValue("");
                        return;
                }
                vtkUnstructuredGrid *ugrid = vtkMimxUnstructuredGridActor::SafeDownCast(this->BBoxList
                        ->GetItem(combobox->GetValueIndex(name)))->GetDataSet();
                vtkActor *actor = this->BBoxList->GetItem(combobox->GetValueIndex(name))->GetActor();
                if(!this->MirrorPlaneWidget)
                {
                        this->MirrorPlaneWidget = vtkPlaneWidget::New();
                        double bounds[6];
                        ugrid->GetBounds(bounds);
                        this->MirrorPlaneWidget->SetInteractor(this->GetMimxMainWindow()->GetRenderWidget()
                                ->GetRenderWindowInteractor());
                        this->MirrorPlaneWidget->SetProp3D(actor);
                        this->MirrorPlaneWidget->PlaceWidget(bounds);
                }
                else
                {
                        double center[3];
                        this->MirrorPlaneWidget->SetEnabled(0);
                        this->MirrorPlaneWidget->GetOrigin(center);
                        this->MirrorPlaneWidget->SetProp3D(actor);
                        this->MirrorPlaneWidget->SetOrigin(center);
                }
                this->MirrorPlaneWidget->SetNormal(1.0,0.0,0.0);
                //this->MirrorPlaneWidget->NormalToXAxisOn();
                this->MirrorPlaneWidget->SetEnabled(1);
        }
}
//----------------------------------------------------------------------------------
void vtkKWMimxEditBBGroup::PlaceMirroringPlaneAboutY()
{
        vtkMimxErrorCallback *callback = this->GetMimxMainWindow()->GetErrorCallback();
        if(!strcmp(this->ObjectListComboBox->GetWidget()->GetValue(),""))
        {
                callback->ErrorMessage("Building Block selection required");
        }
        else
        {
                vtkKWComboBox *combobox = this->ObjectListComboBox->GetWidget();
                const char *name = combobox->GetValue();
                int num = combobox->GetValueIndex(name);
                if(num < 0 || num > combobox->GetNumberOfValues()-1)
                {
                        callback->ErrorMessage("Choose valid Building-block structure");
                        combobox->SetValue("");
                        return;
                }
                vtkUnstructuredGrid *ugrid = vtkMimxUnstructuredGridActor::SafeDownCast(this->BBoxList
                        ->GetItem(combobox->GetValueIndex(name)))->GetDataSet();
                vtkActor *actor = this->BBoxList->GetItem(combobox->GetValueIndex(name))->GetActor();
                if(!this->MirrorPlaneWidget)
                {
                        this->MirrorPlaneWidget = vtkPlaneWidget::New();
                        double bounds[6];
                        ugrid->GetBounds(bounds);
                        this->MirrorPlaneWidget->SetInteractor(this->GetMimxMainWindow()->GetRenderWidget()
                                ->GetRenderWindowInteractor());
                        this->MirrorPlaneWidget->SetProp3D(actor);
                        this->MirrorPlaneWidget->PlaceWidget(bounds);
                }
                else
                {
                        double center[3];
                        this->MirrorPlaneWidget->SetEnabled(0);
                        this->MirrorPlaneWidget->GetOrigin(center);
                        this->MirrorPlaneWidget->SetProp3D(actor);
                        this->MirrorPlaneWidget->SetOrigin(center);
                }
                this->MirrorPlaneWidget->SetNormal(0.0,1.0,0.0);
                this->MirrorPlaneWidget->SetEnabled(1);
        }

}
//----------------------------------------------------------------------------------
void vtkKWMimxEditBBGroup::PlaceMirroringPlaneAboutZ()
{
        vtkMimxErrorCallback *callback = this->GetMimxMainWindow()->GetErrorCallback();
        if(!strcmp(this->ObjectListComboBox->GetWidget()->GetValue(),""))
        {
                callback->ErrorMessage("Building Block selection required");
        }
        else
        {
                vtkKWComboBox *combobox = this->ObjectListComboBox->GetWidget();
                const char *name = combobox->GetValue();
                int num = combobox->GetValueIndex(name);
                if(num < 0 || num > combobox->GetNumberOfValues()-1)
                {
                        callback->ErrorMessage("Choose valid Building-block structure");
                        combobox->SetValue("");
                        return;
                }
                vtkUnstructuredGrid *ugrid = vtkMimxUnstructuredGridActor::SafeDownCast(this->BBoxList
                        ->GetItem(combobox->GetValueIndex(name)))->GetDataSet();
                vtkActor *actor = this->BBoxList->GetItem(combobox->GetValueIndex(name))->GetActor();
                if(!this->MirrorPlaneWidget)
                {
                        this->MirrorPlaneWidget = vtkPlaneWidget::New();
                        double bounds[6];
                        ugrid->GetBounds(bounds);
                        this->MirrorPlaneWidget->SetInteractor(this->GetMimxMainWindow()->GetRenderWidget()
                                ->GetRenderWindowInteractor());
                        this->MirrorPlaneWidget->SetProp3D(actor);
                        this->MirrorPlaneWidget->PlaceWidget(bounds);
                }
                else
                {
                        double center[3];
                        this->MirrorPlaneWidget->SetEnabled(0);
                        this->MirrorPlaneWidget->GetOrigin(center);
                        this->MirrorPlaneWidget->SetProp3D(actor);
                        this->MirrorPlaneWidget->SetOrigin(center);
                }
                this->MirrorPlaneWidget->SetNormal(0.0,0.0,1.0);
                this->MirrorPlaneWidget->SetEnabled(1);
        }
}
//----------------------------------------------------------------------------------
void vtkKWMimxEditBBGroup::RepackMirrorFrame()
{
        this->GetApplication()->Script(
                "pack %s -side top -anchor nw -expand n -padx 2 -pady 2 -after %s", 
                this->MirrorFrame->GetWidgetName(), this->RadioButtonSet->GetWidgetName());

        this->GetApplication()->Script(
                "pack %s -side top -anchor nw -expand y -padx 2 -pady 6", 
                this->TypeOfMirroring->GetWidgetName());

        this->GetApplication()->Script(
                "pack %s -side top -anchor nw -expand n -padx 2 -pady 6", 
                this->AxisSelection->GetWidgetName());
}
