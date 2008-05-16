/*=========================================================================

Program:   MIMX Meshing Toolkit
Module:    $RCSfile: vtkKWMimxEvaluateMeshQualityGroup.cxx,v $
Language:  C++
Date:      $Date: 2008/04/28 02:59:24 $
Version:   $Revision: 1.21 $

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

#include "vtkKWMimxEvaluateMeshQualityGroup.h"
#include "vtkMimxErrorCallback.h"
#include "vtkKWMimxMainNotebook.h"

// VTK includes

#include "vtkCornerAnnotation.h"
#include "vtkImageData.h"
#include "vtkObjectFactory.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkActor.h"
#include "vtkRenderer.h"
#include "vtkUnstructuredGridReader.h"
#include "vtkPlaneWidget.h"
#include "vtkGeometryFilter.h"
#include "vtkProperty.h"
#include "vtkCommand.h"
#include "vtkPlane.h"
#include "vtkRendererCollection.h"
#include "vtkCamera.h"
#include "vtkToolkits.h"
#include <vtksys/SystemTools.hxx>
#include "vtkAnnotatedCubeActor.h"
#include "vtkAppendPolyData.h"
#include "vtkTransformPolyDataFilter.h"
#include "vtkPropAssembly.h"
#include "vtkAxesActor.h"
#include "vtkPolyData.h"
#include "vtkMapper.h"
#include "vtkTransform.h"
#include "vtkCaptionActor2D.h"
#include "vtkTextProperty.h"
#include "vtkWindowToImageFilter.h"
#include "vtkJPEGWriter.h"
#include "vtkUnstructuredGrid.h"
#include "vtkMeshQualityExtended.h"
#include "vtkFieldData.h"
#include "vtkCellData.h"
#include "vtkDoubleArray.h"
#include "vtkIdList.h"


 // KWWidgets includes 

#include "vtkKWApplication.h"
#include "vtkKWFrame.h"
#include "vtkKWFrameWithLabel.h"
#include "vtkKWMenu.h"
#include "vtkKWMenuButton.h"
#include "vtkKWMenuButtonWithSpinButtons.h"
#include "vtkKWMenuButtonWithSpinButtonsWithLabel.h"
#include "vtkKWCheckButtonWithLabel.h"
#include "vtkKWCheckButtonWithChangeColorButton.h"
#include "vtkKWNotebook.h"
#include "vtkKWLoadSaveDialog.h"
#include "vtkKWRenderWidget.h"
#include "vtkKWScale.h"
#include "vtkKWWindow.h"
#include "vtkKWRadioButton.h"
#include "vtkKWRadioButtonSet.h"
#include "vtkKWUserInterfacePanel.h"
#include "vtkKWPushButtonSet.h"
#include "vtkKWPushButton.h"
#include "vtkKWRange.h"
#include "vtkKWThumbWheel.h"
#include "vtkKWListBoxWithScrollbars.h"
#include "vtkOrientationMarkerWidget.h"
#include "vtkKWCheckButtonWithLabel.h"
#include "vtkKWText.h"
#include "vtkKWComboBoxWithLabel.h"
#include "vtkKWComboBox.h"
#include "vtkKWLoadSaveButton.h"
#include "vtkKWMenu.h"
#include "vtkKWEntryWithLabel.h"
#include "vtkKWLabel.h"
#include "vtkKWDialog.h"
#include "vtkKWListBox.h"
#include "vtkKWMimxMainUserInterfacePanel.h"
#include "vtkMimxMeshActor.h"
#include "vtkKWMenuButtonWithLabel.h"


//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkKWMimxEvaluateMeshQualityGroup);
vtkCxxRevisionMacro(vtkKWMimxEvaluateMeshQualityGroup, "$Revision: 1.21 $");

//----------------------------------------------------------------------------



vtkKWMimxEvaluateMeshQualityGroup::vtkKWMimxEvaluateMeshQualityGroup()
{
  this->MeshListComboBox = NULL;
/*
  this->ElementSizeFrame = NULL;
  this->ElementOpacityFrame = NULL;
  this->PlaneControlFrame = NULL;
  this->ReportFrame = NULL;
  this->DisplayFrame = NULL;
  this->SliceScale = NULL; 
  this->OverrideColorButton = NULL;
  this->ElementColorRange = NULL; 
  this->ElementOpacity = NULL; 

  this->PlaneSelectionControls = NULL;
  this->InvertSelector = NULL;
  this->HighlightCellsButton = NULL;
  this->RealTimeWarningButton = NULL;
  this->OutlineSelector = NULL;
  this->ClippedOutlineSelector = NULL;
  this->InteriorOutlineSelector = NULL;
  this->FilledElementSelector = NULL;
  this->HedgehogSelector = NULL;
  this->OrientationCueSelector = NULL;
  this->DecimalDisplayPrecisionWidget = NULL;
  */
  this->NumberOfDistortedEntry = NULL;
  this->NumberOfElementsEntry = NULL;
  this->SummaryFrame = NULL;
  this->QualityMinimumEntry = NULL;
  this->QualityMaximumEntry = NULL;
  this->QualityAverageEntry = NULL;
  this->QualityVarianceEntry = NULL;
  this->SaveButton = NULL;
  this->ViewQualityButton = NULL;
  this->ClippingPlaneMenuButton = NULL;
  this->ViewLegendButton = NULL;
  this->DistortedElementDialog = NULL;
  this->DistortedButtonFrame = NULL;
  this->SaveDistortedButton = NULL;
  this->CancelDistortedButton = NULL;
  this->DistortedElementsReport = NULL;
  this->FileBrowserDialog = NULL;
  this->ButtonFrame = NULL;
  this->QualityTypeButton = NULL;
  this->ViewFrame = NULL;
  this->QualityTypeLabel = NULL;
  this->DistoredListLabel = NULL;
  this->DistortedElementList = vtkIdList::New();
  this->DistortedMeshQuality = vtkDoubleArray::New();
  this->NumberOfCells = 0;
  this->QualityType = MESH_QUALITY_VOLUME;
  this->minimumQuality = 0.0;
  this->maximumQuality = 1.0;
  strcpy(this->meshName, "");
  strcpy(this->qualityName, "");
} 

//----------------------------------------------------------------------------
vtkKWMimxEvaluateMeshQualityGroup::~vtkKWMimxEvaluateMeshQualityGroup()
{
  /* Clean up all allocated Objects */
  if (this->MeshListComboBox)
    this->MeshListComboBox->Delete();
    /*
  if (this->ElementSizeFrame)
    this->ElementSizeFrame->Delete();
  if (this->ElementOpacityFrame)
    this->ElementOpacityFrame->Delete();
  if (this->PlaneControlFrame)
    this->PlaneControlFrame->Delete();
  if (this->ReportFrame)
    this->ReportFrame->Delete();
  if (this->DisplayFrame)
    this->DisplayFrame->Delete();
  if (this->SliceScale)
    this->SliceScale->Delete();
  if (this->OverrideColorButton)
    this->OverrideColorButton->Delete();
  if (this->ElementColorRange)
    this->ElementColorRange->Delete();
  if (this->ElementOpacity)
    this->ElementOpacity->Delete();
  if (this->PlaneSelectionControls)
    this->PlaneSelectionControls->Delete();
  if (this->InvertSelector)
    this->InvertSelector->Delete();
  if (this->HighlightCellsButton)
    this->HighlightCellsButton->Delete();
  if (this->RealTimeWarningButton)
    this->RealTimeWarningButton->Delete();
  if (this->OutlineSelector)
    this->OutlineSelector->Delete();
  if (this->ClippedOutlineSelector)
    this->ClippedOutlineSelector->Delete();
  if (this->InteriorOutlineSelector)
    this->InteriorOutlineSelector->Delete();
  if (this->FilledElementSelector)
    this->FilledElementSelector->Delete();
  if (this->HedgehogSelector)
    this->HedgehogSelector->Delete();
  if (this->OrientationCueSelector)
    this->OrientationCueSelector->Delete();
  if (this->DecimalDisplayPrecisionWidget)
    this->DecimalDisplayPrecisionWidget->Delete();
    */
  if (this->NumberOfDistortedEntry)
    this->NumberOfDistortedEntry->Delete();
  if (this->NumberOfElementsEntry)
    this->NumberOfElementsEntry->Delete();
  if (this->SummaryFrame)
    this->SummaryFrame->Delete();
  if (this->QualityMinimumEntry)
    this->QualityMinimumEntry->Delete();
  if (this->QualityMaximumEntry)
    this->QualityMaximumEntry->Delete();
  if (this->QualityAverageEntry)
    this->QualityAverageEntry->Delete();
  if (this->QualityVarianceEntry)
    this->QualityVarianceEntry->Delete();
  if (this->SaveButton)
    this->SaveButton->Delete();
  if (this->DistortedElementDialog)
    this->DistortedElementDialog->Delete();
  if (this->DistortedButtonFrame)
    this->DistortedButtonFrame->Delete();
  if (this->SaveDistortedButton)
    this->SaveDistortedButton->Delete();
  if (this->CancelDistortedButton)
    this->CancelDistortedButton->Delete();
  if (this->DistortedElementsReport)
    this->DistortedElementsReport->Delete();
  if (this->DistortedElementList)
    this->DistortedElementList->Delete();
  if (this->DistortedMeshQuality)
    this->DistortedMeshQuality->Delete();
  if (this->FileBrowserDialog)
    this->FileBrowserDialog->Delete();
  if (this->ButtonFrame)
    this->ButtonFrame->Delete();
  if (this->ViewQualityButton)
    this->ViewQualityButton->Delete();
  if (this->ClippingPlaneMenuButton)
    this->ClippingPlaneMenuButton->Delete();
  if (this->ViewLegendButton)
    this->ViewLegendButton->Delete();
  if (this->QualityTypeButton)
    this->QualityTypeButton->Delete();
  if (this->ViewFrame)
    this->ViewFrame->Delete();
  if (this->QualityTypeLabel)
    this->QualityTypeLabel->Delete();
  if (this->DistoredListLabel)
    this->DistoredListLabel->Delete();
}

//----------------------------------------------------------------------------
void vtkKWMimxEvaluateMeshQualityGroup::CreateWidget()
{
  
  if (this->IsCreated())
    {
    vtkErrorMacro("class already created");
    return;
    }

  // Call the superclass to create the whole widget
  this->Superclass::CreateWidget();

  this->MainFrame->SetParent(this->GetParent());
  this->MainFrame->Create();
  this->MainFrame->SetLabelText("Mesh Quality");
  this->GetApplication()->Script(
    "pack %s -side top -anchor nw -expand n -padx 2 -pady 0 -fill x", 
    this->MainFrame->GetWidgetName());

  if (!this->MeshListComboBox)  
    this->MeshListComboBox = vtkKWComboBoxWithLabel::New();
  this->MeshListComboBox->SetParent(this->MainFrame->GetFrame());
  this->MeshListComboBox->Create();
  this->MeshListComboBox->SetLabelText("Mesh: ");
  this->MeshListComboBox->SetLabelWidth(15);
  this->MeshListComboBox->GetWidget()->ReadOnlyOn();
  this->MeshListComboBox->GetWidget()->SetBalloonHelpString("Mesh for quality evaluation");

  this->GetApplication()->Script(
    "pack %s -side top -anchor nw -expand y -padx 2 -pady 6 -fill x", 
    this->MeshListComboBox->GetWidgetName());

  /* Quality Metric */
  if (!this->QualityTypeButton) 
    this->QualityTypeButton = vtkKWMenuButtonWithLabel::New();  
  this->QualityTypeButton->SetParent(this->MainFrame->GetFrame());
  this->QualityTypeButton->Create();
  this->QualityTypeButton->SetLabelText("Metric :");
  this->QualityTypeButton->SetLabelWidth(15);
  this->QualityTypeButton->GetWidget()->GetMenu()->DeleteAllItems();
  this->QualityTypeButton->GetWidget()->SetValue("Volume");
  this->QualityTypeButton->GetWidget()->GetMenu()->AddRadioButton("Volume",this, "SetQualityTypeToVolume");
  this->QualityTypeButton->GetWidget()->GetMenu()->AddRadioButton("Edge Collapse",this, "SetQualityTypeToEdgeCollapse");
  this->QualityTypeButton->GetWidget()->GetMenu()->AddRadioButton("Jacobian",this, "SetQualityTypeToJacobian");
  this->QualityTypeButton->GetWidget()->GetMenu()->AddRadioButton("Skew",this, "SetQualityTypeToSkew");
  this->QualityTypeButton->GetWidget()->GetMenu()->AddRadioButton("Angle Out Of Bounds",this, "SetQualityTypeToAngle");
  this->GetApplication()->Script(
    "pack %s -side top -anchor nw -expand y -padx 2 -pady 6 -fill x", 
    this->QualityTypeButton->GetWidgetName());
  
  
  if (!this->ButtonFrame)
    this->ButtonFrame = vtkKWFrame::New();
  this->ButtonFrame->SetParent( this->MainFrame->GetFrame() );
  this->ButtonFrame->Create();
  this->GetApplication()->Script("pack %s -side top -anchor nw -expand n -fill x -padx 2 -pady 2",
              this->ButtonFrame->GetWidgetName() );    

  this->ApplyButton->SetParent(this->ButtonFrame);
  this->ApplyButton->Create();
  this->ApplyButton->SetText("Apply");
  this->ApplyButton->SetCommand(this, "EvaluateMeshQualityApplyCallback");
  this->GetApplication()->Script(
          "pack %s -side left -anchor nw -expand y -padx 10 -pady 6", 
          this->ApplyButton->GetWidgetName());

  this->CancelButton->SetParent(this->ButtonFrame);
  this->CancelButton->Create();
  this->CancelButton->SetText("Cancel");
  this->CancelButton->SetCommand(this, "EvaluateMeshQualityCancelCallback");
  this->GetApplication()->Script(
    "pack %s -side right -anchor ne -expand y -padx 10 -pady 6", 
    this->CancelButton->GetWidgetName());

  if (!this->ViewFrame)
    this->ViewFrame = vtkKWFrameWithLabel::New();
  this->ViewFrame->SetParent( this->MainFrame->GetFrame() );
  this->ViewFrame->Create();
  this->ViewFrame->SetLabelText("View");
  this->GetApplication()->Script("pack %s -side top -anchor nw -expand n -fill x -padx 2 -pady 2",
              this->ViewFrame->GetWidgetName() );    
  this->ViewFrame->CollapseFrame();

/*
  if (!this->NumberOfElementsEntry)
    this->NumberOfElementsEntry = vtkKWEntryWithLabel::New();
  this->NumberOfElementsEntry->SetParent( this->SummaryFrame->GetFrame() );
  this->NumberOfElementsEntry->Create();
  this->NumberOfElementsEntry->GetWidget()->ReadOnlyOn();
  this->NumberOfElementsEntry->GetWidget()->SetWidth(10);
  this->NumberOfElementsEntry->SetLabelText("# Elements");
  this->NumberOfElementsEntry->SetBalloonHelpString("Total number of elements in the mesh");
  this->GetApplication()->Script(
          "grid %s -row 0 -column 0 -sticky ne -padx 2 -pady 6", 
          this->NumberOfElementsEntry->GetWidgetName());

 if (!this->NumberOfDistortedEntry)
    this->NumberOfDistortedEntry = vtkKWEntryWithLabel::New();
  this->NumberOfDistortedEntry->SetParent( this->SummaryFrame->GetFrame() );
  this->NumberOfDistortedEntry->Create();
  this->NumberOfDistortedEntry->GetWidget()->ReadOnlyOn();
  this->NumberOfDistortedEntry->GetWidget()->SetWidth(10);
  this->NumberOfDistortedEntry->SetLabelText("# Distorted");
  this->NumberOfDistortedEntry->SetBalloonHelpString("Number of distorted elements in the mesh");
  this->GetApplication()->Script(
    "grid %s -row 0 -column 1 -sticky ne -padx 2 -pady 6", 
    this->NumberOfDistortedEntry->GetWidgetName());
  
  if (!this->QualityMinimumEntry)
    this->QualityMinimumEntry = vtkKWEntryWithLabel::New();
  this->QualityMinimumEntry->SetParent( this->SummaryFrame->GetFrame() );
  this->QualityMinimumEntry->Create();
  this->QualityMinimumEntry->GetWidget()->ReadOnlyOn();
  this->QualityMinimumEntry->GetWidget()->SetWidth(10);
  this->QualityMinimumEntry->SetLabelText("Minimum");
  this->QualityMinimumEntry->SetBalloonHelpString("Quality minimum value");
  this->GetApplication()->Script(
    "grid %s -row 1 -column 0 -sticky ne -padx 2 -pady 6", 
    this->QualityMinimumEntry->GetWidgetName());
  
  if (!this->QualityMaximumEntry)
    this->QualityMaximumEntry = vtkKWEntryWithLabel::New();
  this->QualityMaximumEntry->SetParent( this->SummaryFrame->GetFrame() );
  this->QualityMaximumEntry->Create();
  this->QualityMaximumEntry->GetWidget()->ReadOnlyOn();
  this->QualityMaximumEntry->GetWidget()->SetWidth(10);
  this->QualityMaximumEntry->SetLabelText("Maximum");
  this->QualityMaximumEntry->SetBalloonHelpString("Quality maximum value");
  this->GetApplication()->Script(
    "grid %s -row 1 -column 1 -sticky ne -padx 2 -pady 6", 
    this->QualityMaximumEntry->GetWidgetName());

    if (!this->QualityAverageEntry)
    this->QualityAverageEntry = vtkKWEntryWithLabel::New();
  this->QualityAverageEntry->SetParent( this->SummaryFrame->GetFrame() );
  this->QualityAverageEntry->Create();
  this->QualityAverageEntry->GetWidget()->ReadOnlyOn();
  this->QualityAverageEntry->GetWidget()->SetWidth(10);
  this->QualityAverageEntry->SetLabelText("Average");
  this->QualityAverageEntry->SetBalloonHelpString("Quality average value");
  this->GetApplication()->Script(
    "grid %s -row 2 -column 0 -sticky ne -padx 2 -pady 6", 
    this->QualityAverageEntry->GetWidgetName());
  
  if (!this->QualityVarianceEntry)
    this->QualityVarianceEntry = vtkKWEntryWithLabel::New();
  this->QualityVarianceEntry->SetParent( this->SummaryFrame->GetFrame() );
  this->QualityVarianceEntry->Create();
  this->QualityVarianceEntry->GetWidget()->ReadOnlyOn();
  this->QualityVarianceEntry->GetWidget()->SetWidth(10);
  this->QualityVarianceEntry->SetLabelText("Variance");
  this->QualityVarianceEntry->SetBalloonHelpString("Quality variance");
  this->GetApplication()->Script(
    "grid %s -row 2 -column 1 -sticky ne -padx 2 -pady 6", 
    this->QualityVarianceEntry->GetWidgetName());
*/
 
  if (!this->ViewQualityButton)
    this->ViewQualityButton = vtkKWCheckButtonWithLabel::New();
  this->ViewQualityButton->SetParent(this->ViewFrame->GetFrame());
  this->ViewQualityButton->Create();
  this->ViewQualityButton->GetWidget()->SetCommand(this, "ViewMeshQualityCallback");
  this->ViewQualityButton->GetWidget()->SetText("View Quality");
  this->ViewQualityButton->GetWidget()->SetEnabled( 0 );
  this->GetApplication()->Script(
        "grid %s -row 0 -column 0 -sticky ne -padx 2 -pady 6", 
        this->ViewQualityButton->GetWidgetName());

  if (!this->ViewLegendButton)
    this->ViewLegendButton = vtkKWCheckButtonWithLabel::New();
  this->ViewLegendButton->SetParent(this->ViewFrame->GetFrame());
  this->ViewLegendButton->Create();
  this->ViewLegendButton->GetWidget()->SetCommand(this, "ViewQualityLegendCallback");
  this->ViewLegendButton->GetWidget()->SetText("View Legend");
  this->ViewLegendButton->GetWidget()->SetEnabled( 0 );
  this->GetApplication()->Script(
        "grid %s -row 0 -column 1 -sticky ne -padx 2 -pady 6", 
        this->ViewLegendButton->GetWidgetName());
  
  if(!this->ClippingPlaneMenuButton)    
                this->ClippingPlaneMenuButton = vtkKWMenuButtonWithLabel::New();
        this->ClippingPlaneMenuButton->SetParent(this->ViewFrame->GetFrame());
        this->ClippingPlaneMenuButton->Create();
        this->ClippingPlaneMenuButton->SetBorderWidth(0);
        this->ClippingPlaneMenuButton->SetReliefToGroove();
        this->ClippingPlaneMenuButton->GetWidget()->SetEnabled( 0 );
        this->ClippingPlaneMenuButton->SetLabelText("Clipping Plane :");
        this->GetApplication()->Script(
          "grid %s -row 1 -column 0 -sticky ne -padx 2 -pady 6", 
                this->ClippingPlaneMenuButton->GetWidgetName());
        this->ClippingPlaneMenuButton->GetWidget()->GetMenu()->AddRadioButton(
                "Off",this, "ClippingPlaneCallback 1");
        this->ClippingPlaneMenuButton->GetWidget()->GetMenu()->AddRadioButton(
                "On",this, "ClippingPlaneCallback 2");
  this->ClippingPlaneMenuButton->GetWidget()->GetMenu()->AddRadioButton(
                "Invert",this, "ClippingPlaneCallback 3");
        this->ClippingPlaneMenuButton->GetWidget()->SetValue("Off");          
 
  if (!this->SaveButton)
    this->SaveButton = vtkKWPushButton::New(); 
  this->SaveButton->SetParent(this->ViewFrame->GetFrame());
  this->SaveButton->Create();
  this->SaveButton->SetText("View Summary");
  this->SaveButton->SetStateToDisabled();
  this->SaveButton->SetCommand(this, "ViewDistortedElemenetsCallback");
  this->GetApplication()->Script(
    "grid %s -row 1 -column 1 -sticky ne -padx 2 -pady 6", 
    this->SaveButton->GetWidgetName());
 
            
#if 0
  //---------------------------------------------------------------------
  //  Element Size Frame
  //---------------------------------------------------------------------
  if (!this->ElementSizeFrame)
    this->ElementSizeFrame = vtkKWFrameWithLabel::New();
  this->ElementSizeFrame->SetParent( this->GetParent() );
  this->ElementSizeFrame->Create();
  this->ElementSizeFrame->SetLabelText("Element Properties");
  this->ElementSizeFrame->CollapseFrame();
  this->GetApplication()->Script("pack %s -side top -anchor nw -expand n -fill x -padx 2 -pady 2",
              this->ElementSizeFrame->GetWidgetName() );    
 
  if (!this->SliceScale)
    this->SliceScale = vtkKWScale::New();
  this->SliceScale->SetParent(this->ElementSizeFrame->GetFrame());
  this->SliceScale->Create();
  this->SliceScale->SetLabelText("Element Size (%)");
  this->SliceScale->SetCommand(this, "SetElementSizeFromScaleCallback");
  
  this->GetApplication()->Script("pack %s -side top -expand n -fill x -padx 2 -pady 2",
                this->SliceScale->GetWidgetName());
    
  if (!this->OverrideColorButton)
    this->OverrideColorButton = vtkKWCheckButtonWithLabel::New();
 
  this->OverrideColorButton->SetParent(this->ElementSizeFrame->GetFrame());
  this->OverrideColorButton->Create();
  this->OverrideColorButton->GetWidget()->SetCommand(this, "SetOverrideColorRangeFromButton");
  this->OverrideColorButton->GetWidget()->SetText("Manually Set Color Range");
  this->GetApplication()->Script("pack %s -side top -expand n -fill x -padx 2 -pady 2",
              this->OverrideColorButton->GetWidgetName());
                 
  if (!this->ElementColorRange) 
        this->ElementColorRange  = vtkKWRange::New();

        this->ElementColorRange->SetParent(this->ElementSizeFrame->GetFrame());
        this->ElementColorRange->Create();
        this->ElementColorRange->SetEnabled( 0 );
        this->ElementColorRange->SetLabelText("A range:");
        this->ElementColorRange->SetWholeRange(-10.0, 40.0);
        this->ElementColorRange->SetRange(0.0, 10.0);
        this->ElementColorRange->SetReliefToGroove();
        this->ElementColorRange->SetBorderWidth(2);
        this->ElementColorRange->SetPadX(2);
        this->ElementColorRange->SetPadY(2);
        this->ElementColorRange->SetCommand(this, "SetColorRangeCallback");
        this->ElementColorRange->SetBalloonHelpString(
    "Specify a subrange of element values here to control how elements are colored. ");
  this->GetApplication()->Script("pack %s -side top -anchor nw -expand n -padx 2 -pady 2 -fill x", 
          this->ElementColorRange->GetWidgetName());


  //---------------------------------------------------------------------
  //  Element Opacity Frame
  //---------------------------------------------------------------------
  if (!this->ElementOpacityFrame)
    this->ElementOpacityFrame = vtkKWFrameWithLabel::New();
  this->ElementOpacityFrame->SetParent( this->GetParent() );
  this->ElementOpacityFrame->Create();
  this->ElementOpacityFrame->SetLabelText("Element Opacity");
  this->ElementOpacityFrame->CollapseFrame();
  this->GetApplication()->Script("pack %s -side top -anchor nw -expand n -fill x -padx 2 -pady 2",
              this->ElementOpacityFrame->GetWidgetName());   
  
  if (!this->ElementOpacity)
    this->ElementOpacity = vtkKWScale::New();
  
  this->ElementOpacity->SetParent( this->ElementOpacityFrame->GetFrame() );
  this->ElementOpacity->Create();
  this->ElementOpacity->SetLabelText("Percent Visible");
  this->ElementOpacity->SetCommand(this, "SetElementAlphaFromScaleCallback");
  this->GetApplication()->Script("pack %s -side top -expand n -fill x -padx 2 -pady 2",
              this->ElementOpacity->GetWidgetName());

  
  //---------------------------------------------------------------------
  //  Plane Control Frame
  //---------------------------------------------------------------------
  if (!this->PlaneControlFrame)
    this->PlaneControlFrame = vtkKWFrameWithLabel::New();
  this->PlaneControlFrame->SetParent( this->GetParent() );
  this->PlaneControlFrame->Create();
  this->PlaneControlFrame->SetLabelText("Selection Plane Controls");
  this->PlaneControlFrame->CollapseFrame();
  this->GetApplication()->Script("pack %s -side top -anchor nw -expand n -fill x -padx 2 -pady 2",
              this->PlaneControlFrame->GetWidgetName());  

  if (!this->PlaneSelectionControls)
    PlaneSelectionControls = vtkKWRadioButtonSet::New();
  this->PlaneSelectionControls->SetParent(PlaneControlFrame->GetFrame());
  this->PlaneSelectionControls->Create();
  this->PlaneSelectionControls->SetPadX(2);
  this->PlaneSelectionControls->SetPadY(2);
  this->PlaneSelectionControls->SetBorderWidth(2);
  this->PlaneSelectionControls->SetReliefToGroove();
  this->PlaneSelectionControls->ExpandWidgetsOn();
  this->PlaneSelectionControls->SetWidgetsPadY(1);
  this->PlaneSelectionControls->SetPackHorizontally(1);
  this->GetApplication()->Script("pack %s -expand n -fill x -padx 2 -pady 2", 
      this->PlaneSelectionControls->GetWidgetName());
    
  vtkKWRadioButton *pb = this->PlaneSelectionControls->AddWidget(1);
  pb->SetText("Enable");
  pb->SetCommand(this, "EnableSelectionPlaneCallback");
        
  pb = this->PlaneSelectionControls->AddWidget(2);
  pb->SetText("Disable");
  pb->SetCommand(this, "ClearSelectionPlaneCallback");
  pb->SelectedStateOn();
              
  if (!this->InvertSelector)
    this->InvertSelector = vtkKWCheckButton::New();
  this->InvertSelector->SetParent(PlaneControlFrame->GetFrame());
  this->InvertSelector->Create();
  this->InvertSelector->SetText("Invert Selection");
  this->InvertSelector->SetCommand(this, "SetInvertSelectionFromButton");
  this->GetApplication()->Script("pack %s -side top -expand n -fill x -padx 2 -pady 2",
              this->InvertSelector->GetWidgetName());


  //---------------------------------------------------------------------
  //  Distorted Elements Frame
  //---------------------------------------------------------------------
  if (!this->ReportFrame)
    this->ReportFrame = vtkKWFrameWithLabel::New();     
  this->ReportFrame->SetParent( this->GetParent() );
  this->ReportFrame->Create();
  this->ReportFrame->SetLabelText("Distorted Elements");
  this->ReportFrame->CollapseFrame();
  this->GetApplication()->Script(
    "pack %s -side top -anchor nw -expand n -fill x -padx 2 -pady 2 ",
    this->ReportFrame->GetWidgetName());   
   
   if (!this->DistortedElementsReport)
     this->DistortedElementsReport = vtkKWListBoxWithScrollbars::New();
   this->DistortedElementsReport->SetParent(this->ReportFrame->GetFrame());
   this->DistortedElementsReport->Create();
   this->DistortedElementsReport->SetBorderWidth(2);
   this->DistortedElementsReport->SetReliefToGroove();
   this->DistortedElementsReport->SetPadX(2);
   this->DistortedElementsReport->SetPadY(2);
   this->DistortedElementsReport->SetWidth(40);
   this->DistortedElementsReport->SetHeight(10);
   this->GetApplication()->Script(
        "pack %s -side top -anchor nw -expand n -fill x -padx 2 -pady 2",
        this->DistortedElementsReport->GetWidgetName()); 
   
  if (!this->HighlightCellsButton)
    this->HighlightCellsButton = vtkKWCheckButtonWithLabel::New();
  this->HighlightCellsButton->SetParent(this->ReportFrame->GetFrame());
  this->HighlightCellsButton->Create();
  this->HighlightCellsButton->GetWidget()->SetCommand(this, "SetHighlightCellsFromButton");
  this->HighlightCellsButton->GetWidget()->SetText("Show only Distorted Elements");
  this->GetApplication()->Script(
        "pack %s -side top -expand n -fill x -padx 2 -pady 2",
        this->HighlightCellsButton->GetWidgetName());
                  
  if (!this->RealTimeWarningButton)
    this->RealTimeWarningButton = vtkKWCheckButtonWithLabel::New();
 
  this->RealTimeWarningButton->SetParent(ReportFrame->GetFrame());
  this->RealTimeWarningButton->Create();
  this->RealTimeWarningButton->GetWidget()->SetCommand(this, "SetRealTimeWarningUpdateFromButton");
  this->RealTimeWarningButton->GetWidget()->SetText("Dynamically Update");
  this->GetApplication()->Script("pack %s -side top -expand n -fill x -padx 2 -pady 2",
              this->RealTimeWarningButton->GetWidgetName());

  //---------------------------------------------------------------------
  //  Display Frame
  //---------------------------------------------------------------------
  this->DisplayFrame = vtkKWFrameWithLabel::New();
  this->DisplayFrame->SetParent( this->GetParent() );
  this->DisplayFrame->Create();
  this->DisplayFrame->SetLabelText("Options");
  this->DisplayFrame->CollapseFrame();
  this->GetApplication()->Script("pack %s -side top -anchor nw -expand n -fill x -padx 2 -pady 2",
              this->DisplayFrame->GetWidgetName());    

  if (!this->OutlineSelector)
    this->OutlineSelector = vtkKWCheckButton::New();
  this->OutlineSelector->SetParent(this->DisplayFrame->GetFrame());
  this->OutlineSelector->SetText("Original Outline");
  this->OutlineSelector->Create();
  this->OutlineSelector->SetCommand(this, "SetOutlineEnableFromButton");
  this->GetApplication()->Script("grid %s -row 0 -column 0 -sticky nw -padx 2 -pady 2",
               this->OutlineSelector->GetWidgetName()); 
                          
  if (!this->ClippedOutlineSelector)
     this->ClippedOutlineSelector = vtkKWCheckButton::New();
  this->ClippedOutlineSelector->SetParent(this->DisplayFrame->GetFrame());
  this->ClippedOutlineSelector->SetText("Clippable Outline");
  this->ClippedOutlineSelector->Create();
  this->ClippedOutlineSelector->SetCommand(this, "SetClippedOutlineEnableFromButton");
  this->GetApplication()->Script("grid %s -row 0 -column 1 -sticky nw -padx 2 -pady 2",
              this->ClippedOutlineSelector->GetWidgetName()); 
               
   //cout << "done clipped outline button" << endl;
                  
  if (!this->InteriorOutlineSelector)
    this->InteriorOutlineSelector = vtkKWCheckButton::New();
  this->InteriorOutlineSelector->SetParent(this->DisplayFrame->GetFrame());
  this->InteriorOutlineSelector->SetText("Interior Outlines");
  this->InteriorOutlineSelector->Create();
  this->InteriorOutlineSelector->SetCommand(this, "SetInteriorOutlineEnableFromButton");
  this->GetApplication()->Script("grid %s -row 1 -column 0 -sticky nw -padx 2 -pady 2",
               this->InteriorOutlineSelector->GetWidgetName());
               
  if (!this->FilledElementSelector)
    this->FilledElementSelector = vtkKWCheckButton::New();  
  this->FilledElementSelector->SetParent(this->DisplayFrame->GetFrame());
  this->FilledElementSelector->SetText( "Filled Elements");
  this->FilledElementSelector->Create();
  this->FilledElementSelector->SetCommand(this, "SetFilledElementEnableFromButton");
  this->GetApplication()->Script("grid %s -row 1 -column 1 -sticky nw -padx 2 -pady 2",
               this->FilledElementSelector->GetWidgetName());
                           
  if (!this->HedgehogSelector)
    this->HedgehogSelector = vtkKWCheckButton::New();
  this->HedgehogSelector->SetParent(this->DisplayFrame->GetFrame());
  this->HedgehogSelector->SetText("Surface Normals");
  this->HedgehogSelector->Create();
  this->HedgehogSelector->SetCommand(this, "SetHedgehogEnableFromButton");
  this->GetApplication()->Script("grid %s -row 2 -column 0 -sticky nw -padx 2 -pady 2",
               this->HedgehogSelector->GetWidgetName());
               
  if (!this->OrientationCueSelector)
    this->OrientationCueSelector = vtkKWCheckButton::New();
  this->OrientationCueSelector->SetParent(this->DisplayFrame->GetFrame());
  this->OrientationCueSelector->SetText("Axis Orientation");
  this->OrientationCueSelector->Create();
  this->OrientationCueSelector->SetCommand(this, "SetOrientationCueEnableFromButton");
  this->GetApplication()->Script("grid %s -row 2 -column 1 -sticky nw -padx 2 -pady 2",
              this->OrientationCueSelector->GetWidgetName()); 

/*
  if (!this->DecimalDisplayPrecisionWidget)
    this->DecimalDisplayPrecisionWidget = vtkKWComboBoxWithLabel::New();

  this->DecimalDisplayPrecisionWidget->SetParent(ReportFrame);
  this->DecimalDisplayPrecisionWidget->Create();
  this->DecimalDisplayPrecisionWidget->SetLabelText("Floating Point Precision: ");
  this->DecimalDisplayPrecisionWidget->GetWidget()->SetCommand(this, "SetDecimalPrecisionCallback");
  char string_i[128];
  for (int i = 0; i < 7 ; i++)
    {
          sprintf(string_i,"%d",i);
          this->DecimalDisplayPrecisionWidget->GetWidget()->AddValue(string_i);
    }
  this->DecimalDisplayPrecisionWidget->GetWidget()->SetValue("3");
    this->GetApplication()->Script(
      "pack %s -side top -anchor nw -expand n -padx 2 -pady 6", 
      this->DecimalDisplayPrecisionWidget->GetWidgetName());
*/
  cout << "done creating widgets" << endl; 
#endif
}

/*
void vtkKWMimxEvaluateMeshQualityGroup::AddOrientationWidget(void)
{
#if 0
          vtkProperty* property;
          
          vtkAnnotatedCubeActor* cube = vtkAnnotatedCubeActor::New();
          cube->SetFaceTextScale( 0.666667 );

          vtkPropCollection* props = vtkPropCollection::New();
          cube->GetActors( props );

          vtkAppendPolyData* append = vtkAppendPolyData::New();

          vtkTransformPolyDataFilter* transformFilter = vtkTransformPolyDataFilter::New();
          vtkTransform* transform = vtkTransform::New();
          transformFilter->SetTransform( transform );

          vtkCollectionSimpleIterator sit;
          props->InitTraversal( sit );
          int nprops = props->GetNumberOfItems();

          for ( int i = 0; i < nprops; i++ )
            {
            vtkActor *node = vtkActor::SafeDownCast( props->GetNextProp( sit ) );

            // the first prop in the collection will be the cube outline, the last
            // will be the text outlines
            //
            if ( node && i == 0 || i == (nprops - 1) )
              {
              vtkPolyData* poly = vtkPolyData::SafeDownCast(node->GetMapper()->GetInput());
              if ( poly )
                {
                transformFilter->SetInput( poly );
                transform->Identity();
                transform->SetMatrix( node->GetMatrix() );
                transform->Scale( 2.0, 2.0, 2.0 );
                transformFilter->Update();

                vtkPolyData* newpoly = vtkPolyData::New();
                newpoly->DeepCopy( transformFilter->GetOutput() );
                append->AddInput( newpoly );
                newpoly->Delete();
                }
              }
            }

          // the orientation marker passed to the widget will be composed of two
          // actors: vtkAxesActor and a vtkAnnotatedCubeActor
          //
          cube->SetFaceTextScale( 0.65 );

          property = cube->GetCubeProperty();
          property->SetColor( 0.5, 1, 1 );

          property = cube->GetTextEdgesProperty();
          property->SetLineWidth( 1 );
          property->SetDiffuse( 0 );
          property->SetAmbient( 1 );
          property->SetColor( 0.1800, 0.2800, 0.2300 );

          // this static function improves the appearance of the text edges
          // since they are overlaid on a surface rendering of the cube's faces
          //
          vtkMapper::SetResolveCoincidentTopologyToPolygonOffset();

          // anatomic labelling
          //
          cube->SetXPlusFaceText ( "X" );
          cube->SetXMinusFaceText( "-x" );
          cube->SetYPlusFaceText ( "Y" );
          cube->SetYMinusFaceText( "-y" );
          cube->SetZPlusFaceText ( "Z" );
          cube->SetZMinusFaceText( "-z" );

          // change the vector text colors
          //
          property = cube->GetXPlusFaceProperty();
          property->SetColor(1, 0, 0);
          property->SetInterpolationToFlat();
          property = cube->GetXMinusFaceProperty();
          property->SetColor(1, 0, 0);
          property->SetInterpolationToFlat();
          property = cube->GetYPlusFaceProperty();
          property->SetColor(0, 1, 0);
          property->SetInterpolationToFlat();
          property = cube->GetYMinusFaceProperty();
          property->SetColor(0, 1, 0);
          property->SetInterpolationToFlat();
          property = cube->GetZPlusFaceProperty();
          property->SetColor(0, 0, 1);
          property->SetInterpolationToFlat();
          property = cube->GetZMinusFaceProperty();
          property->SetColor(0, 0, 1);
          property->SetInterpolationToFlat();

          vtkAxesActor* axes2 = vtkAxesActor::New();

          // simulate a left-handed coordinate system
          //
          transform->Identity();
          //transform->RotateY(90);
          axes2->SetShaftTypeToCylinder();
          axes2->SetUserTransform( transform );
          axes2->SetXAxisLabelText( "X" );
          axes2->SetYAxisLabelText( "Y" );
          axes2->SetZAxisLabelText( "Z" );

          axes2->SetTotalLength( 1.5, 1.5, 1.5 );
          axes2->SetCylinderRadius( 0.500 * axes2->GetCylinderRadius() );
          axes2->SetConeRadius    ( 1.025 * axes2->GetConeRadius() );
          axes2->SetSphereRadius  ( 1.500 * axes2->GetSphereRadius() );

          vtkTextProperty* tprop = axes2->GetXAxisCaptionActor2D()->
            GetCaptionTextProperty();
          tprop->ItalicOn();
          tprop->ShadowOn();
          tprop->SetFontFamilyToTimes();
          tprop->SetColor(0.5,0.5,0.5);

          axes2->GetYAxisCaptionActor2D()->GetCaptionTextProperty()->ShallowCopy( tprop );
          axes2->GetZAxisCaptionActor2D()->GetCaptionTextProperty()->ShallowCopy( tprop );

          // combine orientation markers into one with an assembly
          //
          vtkPropAssembly* assembly = vtkPropAssembly::New();
          assembly->AddPart( axes2 );
          assembly->AddPart( cube );
        
        vtkOrientationMarkerWidget* widget = vtkOrientationMarkerWidget::New();
        widget->SetOutlineColor( 0.9300, 0.5700, 0.1300 );
        widget->SetOrientationMarker( assembly );
        widget->SetInteractor( this->RenderWidget->GetRenderWindowInteractor() );
        widget->SetViewport( 0.0, 0.0, 0.4, 0.4 );
        widget->SetEnabled( 1 );
        widget->InteractiveOff();
        this->SavedOrientationWidget = widget;
        //widget->InteractiveOn();
#endif
}

//----------------------------------------------------------------------------
void vtkKWMimxEvaluateMeshQualityGroup::SetColorRangeCallback(double low, double high)
{
#if 0
        //cout << "color range callback";
        //cout << " low: " << low << " high: " << high << endl;
        this->meshQualityInstance->SetMeshColorRange(low,high);
        this->meshQualityInstance->UpdatePipeline();
        this->RenderWidget->Render();
#endif
}


//----------------------------------------------------------------------------
void vtkKWMimxEvaluateMeshQualityGroup::EnableSelectionPlaneCallback()
{
#if 0
  //cout << "enable selection plane" << endl;
  this->SavedPlaneWidget->PlaceWidget();
  this->SavedPlaneWidget->SetEnabled(1);
  this->meshQualityInstance->EnableCuttingPlane();
#endif
}

//----------------------------------------------------------------------------
void vtkKWMimxEvaluateMeshQualityGroup::ClearSelectionPlaneCallback()
{
#if 0
  //cout << "disable selection plane" << endl;
  this->SavedPlaneWidget->SetEnabled(0);
  this->meshQualityInstance->DisableCuttingPlane();
  this->RenderWidget->Render();
#endif
}


//----------------------------------------------------------------------------
void vtkKWMimxEvaluateMeshQualityGroup::SetThresholdFromScaleCallback(double value)
{
#if 0
  value = value / 100.0;
  this->meshQualityInstance->SetThresholdValue(value);
  this->SavedThresholdValue = value;
  this->meshQualityInstance->UpdatePipeline();
  this->RenderWidget->Render();
#endif
}



//----------------------------------------------------------------------------
void vtkKWMimxEvaluateMeshQualityGroup::SetElementSizeFromScaleCallback(double value)
{
#if 0
  value = value / 100.0;
  this->meshQualityInstance->SetElementShrinkFactor(value);
  this->meshQualityInstance->UpdatePipeline();
  this->RenderWidget->Render();
#endif
}


//----------------------------------------------------------------------------
void vtkKWMimxEvaluateMeshQualityGroup::SetOutlineEnableFromButton(int state)
{
#if 0
  //cout << "outline button callback " << endl;
  this->meshQualityInstance->SetShowOutline(state);
  this->meshQualityInstance->UpdatePipeline();
  this->RenderWidget->Render();
#endif
}


//----------------------------------------------------------------------------
void vtkKWMimxEvaluateMeshQualityGroup::SetClippedOutlineEnableFromButton(int state)
{
#if 0
  //cout << "clipped outline button callback " << endl;
  this->meshQualityInstance->SetShowClippedOutline(state);
  this->meshQualityInstance->UpdatePipeline();
  this->RenderWidget->Render();
#endif
}



//----------------------------------------------------------------------------
void vtkKWMimxEvaluateMeshQualityGroup::SetInteriorOutlineEnableFromButton(int state)
{
#if 0
  //cout << "interior outline button callback " << endl;
  this->meshQualityInstance->SetShowInteriorOutlines(state);
  this->meshQualityInstance->UpdatePipeline();
  this->RenderWidget->Render();
#endif
}


//----------------------------------------------------------------------------
void vtkKWMimxEvaluateMeshQualityGroup::SetFilledElementEnableFromButton(int state)
{
#if 0
  //cout << "filled element button callback " << endl;
  this->meshQualityInstance->SetShowFilledElements(state);
  this->meshQualityInstance->UpdatePipeline();
  this->RenderWidget->GetRenderWindow()->GetRenderers()->GetFirstRenderer()->ResetCamera();
  this->RenderWidget->Render();
  this->Update();
#endif
}  

//----------------------------------------------------------------------------
void vtkKWMimxEvaluateMeshQualityGroup::SetHedgehogEnableFromButton(int state)
{
#if 0
  //cout << "hedgehog button callback " << endl;
  this->meshQualityInstance->SetShowSurfaceHedgehog(state);
  this->meshQualityInstance->UpdatePipeline();
  this->RenderWidget->Render();
#endif
}  

//----------------------------------------------------------------------------
void vtkKWMimxEvaluateMeshQualityGroup::SetHedgehogScale(double value)
{
#if 0
  //cout << "hedgehog scale callback " << endl;
  this->meshQualityInstance->SetHedgehogScale(value);
  this->meshQualityInstance->UpdatePipeline();
  this->RenderWidget->Render();
#endif
}  



//----------------------------------------------------------------------------
void vtkKWMimxEvaluateMeshQualityGroup::SetInvertSelectionFromButton(int state)
{
#if 0
  //cout << "invert button callback " << endl;
  this->meshQualityInstance->SetInvertCuttingPlaneSelection(state);
  this->meshQualityInstance->UpdatePipeline();
  this->RenderWidget->Render();
#endif
}


//----------------------------------------------------------------------------
void vtkKWMimxEvaluateMeshQualityGroup::SetOrientationCueEnableFromButton(int state)
{
#if 0
  //cout << "orientation cue button callback " << endl;
  this->SavedOrientationWidget->SetEnabled(state);
  this->meshQualityInstance->UpdatePipeline();
  this->RenderWidget->Render();
#endif
}


//----------------------------------------------------------------------------
void vtkKWMimxEvaluateMeshQualityGroup::SetOverrideColorRangeFromButton(int state)
{
        double range[2];
#if 0   
  //cout << "override color button callback " << endl;
  this->meshQualityInstance->SetOverrideMeshColorRange(state);
  
  this->ElementColorRange->SetEnabled( state );
  
  if (state)
  {
          // user enabling manual control.  Read the range and update
          // the interface immediately
          this->ElementColorRange->GetRange(range);
          this->SetColorRangeCallback(range[0],range[1]);
  }
  else 
  {
          // called to update the pipeline back to the default colors
          this->meshQualityInstance->UpdatePipeline();
          this->RenderWidget->Render();
  }
#endif
}

//----------------------------------------------------------------------------
void vtkKWMimxEvaluateMeshQualityGroup::SetHighlightCellsFromButton(int state)
{
#if 0   
  if (state)
  {
          this->meshQualityInstance->HighlightFocusElements();
          this->RenderWidget->Render();
  }
  else 
  {
          // called to update the pipeline back to the default colors
          this->meshQualityInstance->CancelElementHighlight();
          this->RenderWidget->Render();
  }
#endif
}

//----------------------------------------------------------------------------
void vtkKWMimxEvaluateMeshQualityGroup::SetRealTimeWarningUpdateFromButton(int state)
{
          
}

//----------------------------------------------------------------------------
void vtkKWMimxEvaluateMeshQualityGroup::SetDecimalPrecisionCallback(char* charValue)
{
#if 0
        if (strlen(charValue))
        {
          //cout << "set display precision to"  << charValue << " places" << endl;
          this->DecimalDisplayPrecision = atoi(charValue);
          this->meshQualityInstance->SetDisplayPrecision(this->DecimalDisplayPrecision);
          this->meshQualityInstance->UpdatePipeline();
          this->RenderWidget->Render();
    }
#endif
}


//----------------------------------------------------------------------------
void vtkKWMimxEvaluateMeshQualityGroup::SetElementAlphaFromScaleCallback(double aFloatValue)
{
#if 0 
  aFloatValue = aFloatValue / 100.0;
  this->meshQualityInstance->SetElementOpacity(aFloatValue);
  this->meshQualityInstance->UpdatePipeline();
  this->RenderWidget->Render();
#endif
}

//----------------------------------------------------------------------------
void vtkKWMimxEvaluateMeshQualityGroup::UpdateRenderingState(void)
{
#if 0
  this->meshQualityInstance->SetThresholdValue(this->SavedThresholdValue);
  this->meshQualityInstance->UpdatePipeline();
  vtkCamera *thisCamera = this->RenderWidget->GetRenderWindow()->GetRenderers()->GetFirstRenderer()->GetActiveCamera();
  thisCamera->Modified();
  this->RenderWidget->GetRenderWindow()->Render();
#endif
}
*/
//----------------------------------------------------------------------------
int vtkKWMimxEvaluateMeshQualityGroup::ViewDistortedElemenetsCallback()
{
  char textValue[128];
  
  if (!this->DistortedElementDialog)
  {
    this->DistortedElementDialog = vtkKWDialog::New();
    this->DistortedElementDialog->SetApplication(this->GetApplication());
    this->DistortedElementDialog->Create();
    this->DistortedElementDialog->ModalOff( );
    this->DistortedElementDialog->SetSize(300, 350);
  }
  sprintf(textValue, "%s Summary", qualityName);
  this->DistortedElementDialog->SetTitle(textValue);
  
  if (!this->SummaryFrame)
  {
    this->SummaryFrame = vtkKWFrame::New();
    this->SummaryFrame->SetParent( this->DistortedElementDialog->GetFrame() );
    this->SummaryFrame->Create();
    this->GetApplication()->Script("pack %s -side top -anchor nw -expand n -fill x -padx 2 -pady 2",
                this->SummaryFrame->GetWidgetName() );    
        }
          
  if (!this->NumberOfElementsEntry)
  {
    this->NumberOfElementsEntry = vtkKWEntryWithLabel::New();
    this->NumberOfElementsEntry->SetParent( this->SummaryFrame );
    this->NumberOfElementsEntry->Create();
    this->NumberOfElementsEntry->GetWidget()->ReadOnlyOn();
    this->NumberOfElementsEntry->GetWidget()->SetWidth(10);
    this->NumberOfElementsEntry->SetLabelText("# Elements");
    this->NumberOfElementsEntry->SetBalloonHelpString("Total number of elements in the mesh");
    this->GetApplication()->Script(
          "grid %s -row 0 -column 0 -sticky ne -padx 2 -pady 6", 
          this->NumberOfElementsEntry->GetWidgetName());
  }
  sprintf(textValue, "%d", this->NumberOfCells);
  this->NumberOfElementsEntry->GetWidget()->SetValue(textValue);

 if (!this->NumberOfDistortedEntry)
 {
    this->NumberOfDistortedEntry = vtkKWEntryWithLabel::New();
    this->NumberOfDistortedEntry->SetParent( this->SummaryFrame );
    this->NumberOfDistortedEntry->Create();
    this->NumberOfDistortedEntry->GetWidget()->ReadOnlyOn();
    this->NumberOfDistortedEntry->GetWidget()->SetWidth(10);
    this->NumberOfDistortedEntry->SetLabelText("# Distorted");
    this->NumberOfDistortedEntry->SetBalloonHelpString("Number of distorted elements in the mesh"); 
    this->GetApplication()->Script(
      "grid %s -row 0 -column 1 -sticky ne -padx 2 -pady 6", 
    this->NumberOfDistortedEntry->GetWidgetName());
  }
  sprintf(textValue, "%d", this->DistortedElementList->GetNumberOfIds());
  this->NumberOfDistortedEntry->GetWidget()->SetValue(textValue);
  
  if (!this->QualityMinimumEntry)
  {
    this->QualityMinimumEntry = vtkKWEntryWithLabel::New();
    this->QualityMinimumEntry->SetParent( this->SummaryFrame );
    this->QualityMinimumEntry->Create();
    this->QualityMinimumEntry->GetWidget()->ReadOnlyOn();
    this->QualityMinimumEntry->GetWidget()->SetWidth(10);
    this->QualityMinimumEntry->SetLabelText("Minimum");
    this->QualityMinimumEntry->SetBalloonHelpString("Quality minimum value");
    this->GetApplication()->Script(
      "grid %s -row 1 -column 0 -sticky ne -padx 2 -pady 6", 
      this->QualityMinimumEntry->GetWidgetName());
  }
  sprintf(textValue, "%6.3f", this->minimumQuality);
  this->QualityMinimumEntry->GetWidget()->SetValue(textValue);
  
  if (!this->QualityMaximumEntry)
  {
    this->QualityMaximumEntry = vtkKWEntryWithLabel::New();
    this->QualityMaximumEntry->SetParent( this->SummaryFrame );
    this->QualityMaximumEntry->Create();
    this->QualityMaximumEntry->GetWidget()->ReadOnlyOn();
    this->QualityMaximumEntry->GetWidget()->SetWidth(10);
    this->QualityMaximumEntry->SetLabelText("Maximum");
    this->QualityMaximumEntry->SetBalloonHelpString("Quality maximum value");  
    this->GetApplication()->Script(
      "grid %s -row 1 -column 1 -sticky ne -padx 2 -pady 6", 
      this->QualityMaximumEntry->GetWidgetName());
  }
  sprintf(textValue, "%6.3f", this->maximumQuality );
  this->QualityMaximumEntry->GetWidget()->SetValue(textValue);
  
  if (!this->QualityAverageEntry)
  {
    this->QualityAverageEntry = vtkKWEntryWithLabel::New();
    this->QualityAverageEntry->SetParent( this->SummaryFrame );
    this->QualityAverageEntry->Create();
    this->QualityAverageEntry->GetWidget()->ReadOnlyOn();
    this->QualityAverageEntry->GetWidget()->SetWidth(10);
    this->QualityAverageEntry->SetLabelText("Average");
    this->QualityAverageEntry->SetBalloonHelpString("Quality average value");  
    this->GetApplication()->Script(
      "grid %s -row 2 -column 0 -sticky ne -padx 2 -pady 6", 
      this->QualityAverageEntry->GetWidgetName());
  }
  sprintf(textValue, "%6.3f", this->averageQuality);
  this->QualityAverageEntry->GetWidget()->SetValue(textValue);
  
  
  if (!this->QualityVarianceEntry)
  {
    this->QualityVarianceEntry = vtkKWEntryWithLabel::New();
    this->QualityVarianceEntry->SetParent( this->SummaryFrame );
    this->QualityVarianceEntry->Create();
    this->QualityVarianceEntry->GetWidget()->ReadOnlyOn();
    this->QualityVarianceEntry->GetWidget()->SetWidth(10);
    this->QualityVarianceEntry->SetLabelText("Variance");
    this->QualityVarianceEntry->SetBalloonHelpString("Quality variance");
    this->GetApplication()->Script(
      "grid %s -row 2 -column 1 -sticky ne -padx 2 -pady 6", 
      this->QualityVarianceEntry->GetWidgetName());
  }
  sprintf(textValue, "%6.3f", this->varianceQuality );
  this->QualityVarianceEntry->GetWidget()->SetValue(textValue);  
  
  if (!this->DistoredListLabel)
  {
    this->DistoredListLabel = vtkKWLabel::New();
    this->DistoredListLabel->SetParent( this->DistortedElementDialog->GetFrame() );
    this->DistoredListLabel->Create( );
    this->DistoredListLabel->SetText("Distorted Elements");
    this->GetApplication()->Script(
          "pack %s -side top -anchor center -expand n -fill x -padx 2 -pady 2", 
          this->DistoredListLabel->GetWidgetName());
        }
          
  if (!this->DistortedElementsReport)
  {
    this->DistortedElementsReport = vtkKWListBoxWithScrollbars::New();
    this->DistortedElementsReport->SetParent(this->DistortedElementDialog->GetFrame());
    this->DistortedElementsReport->Create();
    this->DistortedElementsReport->SetBorderWidth(2);
    this->DistortedElementsReport->SetReliefToGroove();
    this->DistortedElementsReport->SetPadX(2);
    this->DistortedElementsReport->SetPadY(2);
    this->DistortedElementsReport->SetWidth(80);
    this->DistortedElementsReport->SetHeight(40);
    this->GetApplication()->Script(
      "pack %s -side top -anchor nw -expand n -fill x -padx 2 -pady 2",
      this->DistortedElementsReport->GetWidgetName()); 
  }
  
  if (!this->DistortedButtonFrame)
  {
    this->DistortedButtonFrame = vtkKWFrame::New();
    this->DistortedButtonFrame->SetParent( this->DistortedElementDialog->GetFrame() );
    this->DistortedButtonFrame->Create();
    this->GetApplication()->Script("pack %s -side top -anchor nw -expand n -fill x -padx 2 -pady 6",
                this->DistortedButtonFrame->GetWidgetName() );    
  }
  
  if (!this->SaveDistortedButton)
  {
    this->SaveDistortedButton = vtkKWPushButton::New();
    this->SaveDistortedButton->SetParent( this->DistortedButtonFrame );
    this->SaveDistortedButton->Create();
    this->SaveDistortedButton->SetText("Save");
    this->SaveDistortedButton->SetWidth(10);
    this->SaveDistortedButton->SetCommand(this, "DistortedElementDialogSaveCallback");
    this->GetApplication()->Script("pack %s -side left -anchor nw -expand n -fill x -padx 10 -pady 6",
                this->SaveDistortedButton->GetWidgetName() ); 
  }
  
  if (!this->CancelDistortedButton)
  {
    this->CancelDistortedButton = vtkKWPushButton::New();
    this->CancelDistortedButton->SetParent( this->DistortedButtonFrame );
    this->CancelDistortedButton->Create();
    this->CancelDistortedButton->SetWidth(10);
    this->CancelDistortedButton->SetText("Cancel");
    this->CancelDistortedButton->SetCommand(this, "DistortedElementDialogCancelCallback");
    this->GetApplication()->Script("pack %s -side right -anchor ne -expand n -fill x -padx 10 -pady 6",
                this->CancelDistortedButton->GetWidgetName() ); 
  }
  this->DistortedElementsReport->GetWidget()->DeleteAll();

  char formatstring[80];
  //sprintf(formatstring, "Elements: %d, Distorted: %d", this->NumberOfCells, this->DistortedElementList->GetNumberOfIds());
  //this->DistortedElementsReport->GetWidget()->Append(formatstring);
  //std::cout << formatstring << std::endl;
  
  for (int i=0;i<this->DistortedElementList->GetNumberOfIds();i++)
  {
    vtkIdType thisId = this->DistortedElementList->GetId(i);
    double thisQ = this->DistortedMeshQuality->GetValue(i);
    sprintf(formatstring, "ElementID: %06d, quality: %6.3f", thisId, thisQ);
    //std::cout << formatstring << std::endl;
    this->DistortedElementsReport->GetWidget()->Append(formatstring);
  }
     
  this->DistortedElementDialog->Invoke();
  return 1;
}

//----------------------------------------------------------------------------
int vtkKWMimxEvaluateMeshQualityGroup::DistortedElementDialogCancelCallback()
{
  this->DistortedElementDialog->Cancel();
  return 1;
}

//----------------------------------------------------------------------------
int vtkKWMimxEvaluateMeshQualityGroup::DistortedElementDialogSaveCallback()
{

  if(!this->FileBrowserDialog)
        {
                this->FileBrowserDialog = vtkKWLoadSaveDialog::New() ;
                this->FileBrowserDialog->SaveDialogOn();
                this->FileBrowserDialog->SetApplication(this->GetApplication());
                this->FileBrowserDialog->Create();
                this->FileBrowserDialog->SetTitle ("Save Distorted Elements");
                this->FileBrowserDialog->SetFileTypes ("{{CSV files} {.csv}}");
                this->FileBrowserDialog->SetDefaultExtension (".csv");
        }
        this->FileBrowserDialog->Invoke();
        if(this->FileBrowserDialog->GetStatus() == vtkKWDialog::StatusOK)
        {
                if(this->FileBrowserDialog->GetFileName())
                {
                        const char *filename = this->FileBrowserDialog->GetFileName();
                        ofstream outFile;
      outFile.open(filename);
      if (outFile.fail()) 
      {
        vtkMimxErrorCallback *callback = this->GetMimxMainWindow()->GetErrorCallback();
        callback->ErrorMessage("Unable to open the specified file. Check permissions.");
        return 0;
      }
      char formatstring[80];
      sprintf(formatstring, "Metric: %s", this->qualityName);
      outFile << formatstring << std::endl;
      sprintf(formatstring, "Elements: %d, Distorted: %d", this->NumberOfCells, this->DistortedElementList->GetNumberOfIds());
      outFile << formatstring << std::endl;
      sprintf(formatstring, "Minimum: %6.3f", this->minimumQuality);
      outFile << formatstring << std::endl;
      sprintf(formatstring, "Maximum: %6.3f", this->maximumQuality);
      outFile << formatstring << std::endl;
      sprintf(formatstring, "Average: %6.3f", this->averageQuality);
      outFile << formatstring << std::endl;
      sprintf(formatstring, "Variance: %6.3f", this->varianceQuality);
      outFile << formatstring << std::endl;
      
      for (int i=0;i<this->DistortedElementList->GetNumberOfIds();i++)
      {
        vtkIdType thisId = this->DistortedElementList->GetId(i);
        double thisQ = this->DistortedMeshQuality->GetValue(i);
        sprintf(formatstring, "ElementID: %06d, quality: %6.3f", thisId, thisQ);
        outFile << formatstring << std::endl;
      }
      outFile.close();
                        
                        return 1;
                }
        }
  
  return 0;
}


//----------------------------------------------------------------------------
int vtkKWMimxEvaluateMeshQualityGroup::EvaluateMeshQualityCancelCallback()
{
  this->GetApplication()->Script("pack forget %s", this->MainFrame->GetWidgetName());
  this->MenuGroup->SetMenuButtonsEnabled(1);
  this->GetMimxMainWindow()->GetMainUserInterfacePanel()->GetMimxMainNotebook()->SetEnabled(1);
  
  vtkKWComboBox *combobox = this->MeshListComboBox->GetWidget();
  
  if ( strlen(this->meshName) )
  {    
    vtkMimxMeshActor *meshActor = vtkMimxMeshActor::SafeDownCast(
             this->FEMeshList->GetItem(combobox->GetValueIndex( this->meshName )));     
    meshActor->SetMeshScalarVisibility(false);
    meshActor->SetMeshLegendVisibility(false);
    meshActor->DisableMeshCuttingPlane( );
  }
  this->GetMimxMainWindow()->GetRenderWidget()->Render(); 
  strcpy(this->meshName, "");  
  strcpy(this->qualityName, "");
  return 1;
}

//----------------------------------------------------------------------------
int vtkKWMimxEvaluateMeshQualityGroup::EvaluateMeshQualityApplyCallback()
{
  vtkMimxErrorCallback *callback = this->GetMimxMainWindow()->GetErrorCallback();
  
  if(!strcmp(this->MeshListComboBox->GetWidget()->GetValue(),""))
  {
        callback->ErrorMessage("FE Mesh not chosen");
        return 0;
  }
  
  vtkKWComboBox *combobox = this->MeshListComboBox->GetWidget();
  strcpy(this->meshName, combobox->GetValue());
  int num = combobox->GetValueIndex( this->meshName );
  if(num < 0 || num > combobox->GetNumberOfValues()-1)
  {
    callback->ErrorMessage("Choose valid FE mesh");
    combobox->SetValue("");
    return 0;
  }

  vtkUnstructuredGrid *ugrid = vtkMimxMeshActor::SafeDownCast(
           this->FEMeshList->GetItem(combobox->GetValueIndex(this->meshName)))->GetDataSet();
  
  vtkMeshQualityExtended *meshQualityFilter = vtkMeshQualityExtended::New();
  meshQualityFilter->SetInput( ugrid );
  int qualityMetric = 1;
  std::string metricFieldName;
  switch ( this->QualityType )
  {
    case MESH_QUALITY_VOLUME: 
      meshQualityFilter->SetHexQualityMeasureToVolume( ); 
      metricFieldName = "Volume";
      break;
    case MESH_QUALITY_EDGE: 
      meshQualityFilter->SetHexQualityMeasureToEdgeCollapse( ); 
      metricFieldName = "Edge Collapse";
      break;
    case MESH_QUALITY_JACOBIAN: 
      meshQualityFilter->SetHexQualityMeasureToJacobian( ); 
      metricFieldName = "Jacobian";
      break;
    case MESH_QUALITY_SKEW: 
      meshQualityFilter->SetHexQualityMeasureToSkew( ); 
      metricFieldName = "Skew";
      break;
    case MESH_QUALITY_ANGLE: 
      meshQualityFilter->SetHexQualityMeasureToAngleOutOfBounds( ); 
      metricFieldName = "Angle Out of Bounds";
      break;
    default:
      callback->ErrorMessage("Invalid Metric Specified");
      return 0;
  }
  meshQualityFilter->Update( );
  
  strcpy(this->qualityName, metricFieldName.c_str());
  //char textValue[128];
  /*
  sprintf(textValue, "%d", ugrid->GetNumberOfCells() );
  this->NumberOfElementsEntry->GetWidget()->SetValue(textValue);
  */
  this->minimumQuality = meshQualityFilter->GetOutput()->GetFieldData()->
        GetArray( "Mesh Hexahedron Quality" )->GetComponent( 0, 0 );   
  this->maximumQuality = meshQualityFilter->GetOutput()->GetFieldData()->
        GetArray( "Mesh Hexahedron Quality" )->GetComponent( 0, 2 );
  this->averageQuality = meshQualityFilter->GetOutput()->GetFieldData()->
         GetArray( "Mesh Hexahedron Quality" )->GetComponent( 0, 1 );
  this->varianceQuality = meshQualityFilter->GetOutput()->GetFieldData()->
         GetArray( "Mesh Hexahedron Quality" )->GetComponent( 0, 3 );
  /*             
  sprintf(textValue, "%6.3f", this->minimumQuality);
  this->QualityMinimumEntry->GetWidget()->SetValue(textValue);
  
  sprintf(textValue, "%6.3f", this->maximumQuality );
  this->QualityMaximumEntry->GetWidget()->SetValue(textValue);
       
  sprintf(textValue, "%6.3f", this->averageQuality);
  this->QualityAverageEntry->GetWidget()->SetValue(textValue);

  sprintf(textValue, "%6.3f", this->varianceQuality );
  this->QualityVarianceEntry->GetWidget()->SetValue(textValue);
  */
  
  int badValueCount = 0;
  vtkUnstructuredGrid *mesh = (vtkUnstructuredGrid*) meshQualityFilter->GetOutput();
  this->DistortedElementList->Initialize();
  this->DistortedMeshQuality->Initialize();
  this->NumberOfCells = ugrid->GetNumberOfCells();
  
  for (int i=0; i< ugrid->GetNumberOfCells() ; i++) 
  {
    double thisQ = ((vtkDoubleArray*)(mesh->GetCellData())->GetArray("Quality"))->GetValue(i);

    // negative value is significant, so lets start snooping on values
    if (thisQ < 0) 
    {
        badValueCount++;

      int thisID;
      if (mesh->GetCellData()->GetArray("ELEMENT_ID") == NULL)
      {
        thisID = i;
      }
      else
      {
        thisID = ((vtkIntArray*)mesh->GetCellData()->GetArray("ELEMENT_ID"))->GetValue(i);
      }  
      
      this->DistortedElementList->InsertNextId( thisID );
      this->DistortedMeshQuality->InsertNextValue( thisQ );
    }
  } 
  /*
  sprintf(textValue, "%d", badValueCount );
  this->NumberOfDistortedEntry->GetWidget()->SetValue(textValue);
  */
  vtkDoubleArray *qualityArray = (vtkDoubleArray*)(mesh->GetCellData())->GetArray("Quality");
  qualityArray->SetName( metricFieldName.c_str() );
  ugrid->GetCellData()->RemoveArray( metricFieldName.c_str() );
  ugrid->GetCellData()->AddArray( qualityArray );

  this->SaveButton->SetEnabled( 1 );
  this->ClippingPlaneMenuButton->GetWidget()->SetEnabled( 1 );
  this->ViewQualityButton->GetWidget()->SetEnabled( 1 );
  this->ViewLegendButton->GetWidget()->SetEnabled( 1 );
  this->ViewFrame->ExpandFrame();
  meshQualityFilter->Delete();
  
  vtkMimxMeshActor *meshActor = vtkMimxMeshActor::SafeDownCast(
           this->FEMeshList->GetItem(combobox->GetValueIndex(this->meshName)));
  meshActor->SetLegendRange(this->minimumQuality, this->maximumQuality);
  
  this->ViewQualityButton->GetWidget()->Select( );
  this->ViewLegendButton->GetWidget()->Select( );
  
  if (this->ViewQualityButton->GetWidget()->GetSelectedState( ) )
  {
    meshActor->SetMeshScalarName( qualityName );
    this->GetMimxMainWindow()->GetRenderWidget()->Render();
  }
           
  this->GetMimxMainWindow()->SetStatusText("Evaluated Mesh Quality");
  
  return 1;      
}

//----------------------------------------------------------------------------
void vtkKWMimxEvaluateMeshQualityGroup::Update()
{
        this->UpdateEnableState();
}

//---------------------------------------------------------------------------
void vtkKWMimxEvaluateMeshQualityGroup::UpdateEnableState()
{
        this->UpdateObjectLists();
        this->Superclass::UpdateEnableState();
}
//----------------------------------------------------------------------------
void vtkKWMimxEvaluateMeshQualityGroup::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}

//------------------------------------------------------------------------------
void vtkKWMimxEvaluateMeshQualityGroup::UpdateObjectLists()
{
  this->MeshListComboBox->GetWidget()->DeleteAllValues();
  
  int defaultItem = -1;
  for (int i = 0; i < this->FEMeshList->GetNumberOfItems(); i++)
  {
    this->MeshListComboBox->GetWidget()->AddValue(
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
    this->MeshListComboBox->GetWidget()->SetValue(
          this->FEMeshList->GetItem(defaultItem)->GetFileName());
  }
  
}

//------------------------------------------------------------------------------
void vtkKWMimxEvaluateMeshQualityGroup::ClearStatsEntry()
{
  /*
  this->NumberOfElementsEntry->GetWidget()->SetValue("");
  this->QualityMinimumEntry->GetWidget()->SetValue("");
  this->QualityMaximumEntry->GetWidget()->SetValue("");
  this->QualityAverageEntry->GetWidget()->SetValue("");
  this->QualityVarianceEntry->GetWidget()->SetValue("");
  this->NumberOfDistortedEntry->GetWidget()->SetValue("");
  */ 
  this->ClippingPlaneMenuButton->GetWidget()->SetEnabled( 0 );
  this->ViewQualityButton->GetWidget()->SetEnabled( 0 );
  this->ViewLegendButton->GetWidget()->SetEnabled( 0 );
  this->SaveButton->SetEnabled( 0 );
  
  this->ClippingPlaneMenuButton->GetWidget()->SetValue("Off");
  this->ViewQualityButton->GetWidget()->Deselect( );
  this->ViewLegendButton->GetWidget()->Deselect( );
  this->ViewFrame->CollapseFrame();
}

//------------------------------------------------------------------------------
int vtkKWMimxEvaluateMeshQualityGroup::ViewMeshQualityCallback(int mode)
{
  std::string metricFieldName;
  switch ( this->QualityType )
  {
    case MESH_QUALITY_VOLUME: 
      metricFieldName = "Volume";
      break;
    case MESH_QUALITY_EDGE: 
      metricFieldName = "Edge Collapse";
      break;
    case MESH_QUALITY_JACOBIAN: 
      metricFieldName = "Jacobian";
      break;
    case MESH_QUALITY_SKEW: 
      metricFieldName = "Skew";
      break;
    case MESH_QUALITY_ANGLE: 
      metricFieldName = "Angle Out of Bounds";
      break;
    default:
      vtkMimxErrorCallback *callback = this->GetMimxMainWindow()->GetErrorCallback();
      callback->ErrorMessage("Invalid Metric Specified");
      return 0;
  }

  this->ViewLegendButton->GetWidget()->Select( );
  
  vtkKWComboBox *combobox = this->MeshListComboBox->GetWidget();
  
  if ( strlen(this->meshName) )
  {
    vtkMimxMeshActor *meshActor = vtkMimxMeshActor::SafeDownCast(
             this->FEMeshList->GetItem(combobox->GetValueIndex( this->meshName )));
             
    if ( mode )
    {
      meshActor->SetLegendTextColor(this->GetMimxMainWindow()->GetTextColor());
      meshActor->SetMeshScalarName( qualityName );
      meshActor->SetMeshScalarVisibility(true);
      //this->ViewQualityLegendCallback(1);
      this->ViewLegendButton->GetWidget()->Select( );
      this->GetMimxMainWindow()->GetRenderWidget()->Render();
    }
    else
    {
      meshActor->SetMeshScalarVisibility(false);
      this->ViewQualityLegendCallback(0);
      this->ViewLegendButton->GetWidget()->SetSelectedState( 0 );
      //this->ViewQualityButton->GetWidget()->Deselect()
      this->GetMimxMainWindow()->GetRenderWidget()->Render(); 
    }
  }
 return 1; 
}

//------------------------------------------------------------------------------
int vtkKWMimxEvaluateMeshQualityGroup::ClippingPlaneCallback(int mode)
{
  vtkKWComboBox *combobox = this->MeshListComboBox->GetWidget();
  vtkMimxMeshActor *meshActor = vtkMimxMeshActor::SafeDownCast(
           this->FEMeshList->GetItem(combobox->GetValueIndex( this->meshName )));
           
  if (mode == 1)
  {
    meshActor->DisableMeshCuttingPlane();
  }
  else if (mode == 2)
  {
    meshActor->EnableMeshCuttingPlane();
    meshActor->SetInvertCuttingPlane( false );
  }
  else
  {
    meshActor->EnableMeshCuttingPlane();
    meshActor->SetInvertCuttingPlane( true );  
  }
  this->GetMimxMainWindow()->GetRenderWidget()->Render(); 
  return 1;
}

//------------------------------------------------------------------------------
int vtkKWMimxEvaluateMeshQualityGroup::ViewQualityLegendCallback(int mode)
{
  vtkKWComboBox *combobox = this->MeshListComboBox->GetWidget();
  
  if ( strlen(this->meshName) )
  {
    vtkMimxMeshActor *meshActor = vtkMimxMeshActor::SafeDownCast(
             this->FEMeshList->GetItem(combobox->GetValueIndex( this->meshName )));
             
    if ( mode )
    {
      meshActor->SetMeshLegendVisibility(true);
      this->GetMimxMainWindow()->GetRenderWidget()->Render();
    }
    else
    {
      meshActor->SetMeshLegendVisibility(false);
      this->GetMimxMainWindow()->GetRenderWidget()->Render(); 
    }
  }
  return 1;
}

