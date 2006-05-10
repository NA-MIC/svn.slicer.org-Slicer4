/*=========================================================================

  Module:    $RCSfile: vtkKWWindowLevelThresholdEditor.cxx,v $

  Copyright (c) Kitware, Inc.
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkKWWindowLevelThresholdEditor.h"

#include "vtkObjectFactory.h"

#include "vtkKWMenuButton.h"
#include "vtkKWFrame.h"
#include "vtkKWMenu.h"

//----------------------------------------------------------------------------
vtkStandardNewMacro( vtkKWWindowLevelThresholdEditor );
vtkCxxRevisionMacro(vtkKWWindowLevelThresholdEditor, "$Revision: 1.49 $");

//----------------------------------------------------------------------------
vtkKWWindowLevelThresholdEditor::vtkKWWindowLevelThresholdEditor()
{
  this->Command = NULL;
  //this->StartCommand = NULL;
  //this->EndCommand   = NULL;

  this->ImageData = NULL;

  this->WindowLevelAutoManual = vtkKWMenuButtonWithLabel::New() ;
  this->TresholdAutoManual = vtkKWMenuButtonWithLabel::New();
  this->TresholdApply = vtkKWCheckButtonWithLabel::New();

  this->WindoLevelRange = vtkKWRange::New();
  this->ThresholdRange = vtkKWRange::New();
  this->ColorTransferFunctionEditor = vtkKWColorTransferFunctionEditor::New();   
  this->Histogram = vtkKWHistogram::New();
  this->TransferFunction = vtkColorTransferFunction::New();
  this->ColorTransferFunctionEditor->SetColorTransferFunction(this->TransferFunction);
  this->TransferFunction->Delete();
}

//----------------------------------------------------------------------------
vtkKWWindowLevelThresholdEditor::~vtkKWWindowLevelThresholdEditor()
{
  if (this->Command)
    {
    delete [] this->Command;
    }
  if (this->ImageData)
    {
    this->SetImageData(NULL);
    }
  this->WindowLevelAutoManual->Delete();
  this->TresholdAutoManual->Delete();
  this->TresholdApply->Delete();
  this->WindoLevelRange->Delete();
  this->ThresholdRange->Delete();
  this->Histogram->Delete();
  this->ColorTransferFunctionEditor->Delete();

}

void vtkKWWindowLevelThresholdEditor::SetImageData(vtkImageData* imageData)
{
  if (this->ImageData != imageData) 
    {
    vtkImageData* tempImageData = this->ImageData;
    this->ImageData = imageData;
    if (this->ImageData != NULL)
      {
      this->ImageData->Register(this);
      }
    if (tempImageData != NULL)
      {
      tempImageData->UnRegister(this);
      }
    this->Modified();   
      
    this->UpdateFromImage();

    this->UpdateTransferFunction();
    }
}


void vtkKWWindowLevelThresholdEditor::SetWindowLevel(double window, double level)
{
  
  this->WindoLevelRange->SetRange(level - 0.5*window, level + 0.5*window);
  
  this->UpdateTransferFunction();
}
double vtkKWWindowLevelThresholdEditor::GetWindow()
{
  double *range = this->WindoLevelRange->GetRange();
  return range[1] -  range[0];
}

double vtkKWWindowLevelThresholdEditor::GetLevel()
{
  double *range = this->WindoLevelRange->GetRange();
  return 0.5 * (range[1] +  range[0]);
}

 
void vtkKWWindowLevelThresholdEditor::SetThreshold(double lower, double upper)
{
  this->ThresholdRange->SetWholeRange(lower, upper);
  this->ThresholdRange->SetRange(lower, upper);
  this->UpdateTransferFunction();
}

double vtkKWWindowLevelThresholdEditor::GetLowerThreshold()
{
  double *range = this->ThresholdRange->GetRange();
  return range[0];
}


double vtkKWWindowLevelThresholdEditor::GetUpperThreshold()
{
  double *range = this->ThresholdRange->GetRange();
  return range[1];
}



//----------------------------------------------------------------------------
void vtkKWWindowLevelThresholdEditor::CreateWidget()
{
  // Check if already created
  
  if (this->IsCreated())
    {
    vtkErrorMacro(<< this->GetClassName() << " already created");
    return;
    }
  
  // Call the superclass to create the whole widget
  this->Superclass::CreateWidget();
  
  this->UpdateTransferFunction();  

  vtkKWFrame *winLevelFrame = vtkKWFrame::New ( );
  winLevelFrame->SetParent (this);
  winLevelFrame->Create();
  this->Script(
    "pack %s -side top -anchor nw -expand n -padx 2 -pady 2", 
    winLevelFrame->GetWidgetName());

  this->WindowLevelAutoManual->SetParent(winLevelFrame);
  this->WindowLevelAutoManual->Create();
  this->WindowLevelAutoManual->GetWidget()->GetMenu()->AddRadioButton ( "Auto" );
  this->WindowLevelAutoManual->GetWidget()->GetMenu()->AddRadioButton ( "Manual" );
  this->WindowLevelAutoManual->SetLabelText("Window/Level:");
  this->WindowLevelAutoManual->GetWidget()->SetValue ( "Auto" );
  this->Script(
    "pack %s -side left -anchor nw -expand n -padx 2 -pady 2", 
    this->WindowLevelAutoManual->GetWidgetName());

  this->WindoLevelRange->SetParent(winLevelFrame);
  this->WindoLevelRange->Create();
  this->WindoLevelRange->SymmetricalInteractionOn();
  this->WindoLevelRange->SetCommand(this, "ProcessWindowLevelCommand");
  this->WindoLevelRange->SetStartCommand(this, "ProcessWindowLevelStartCommand");
  this->Script(
    "pack %s -side left -anchor nw -expand yes -padx 2 -pady 2", 
    this->WindoLevelRange->GetWidgetName());
  
  vtkKWFrame *threshFrame = vtkKWFrame::New ( );
  threshFrame->SetParent (this);
  threshFrame->Create();
  this->Script(
    "pack %s -side top -anchor nw -expand n -padx 2 -pady 2", 
    threshFrame->GetWidgetName());

  this->TresholdAutoManual->SetParent(threshFrame);
  this->TresholdAutoManual->Create();
  this->TresholdAutoManual->GetWidget()->GetMenu()->AddRadioButton ( "Auto" );
  this->TresholdAutoManual->GetWidget()->GetMenu()->AddRadioButton ( "Manual" );
  this->TresholdAutoManual->GetWidget()->SetValue ( "Auto" );
  this->TresholdAutoManual->SetLabelText("Threshold:");
  this->Script(
    "pack %s -side left -anchor nw -expand n -padx 2 -pady 2", 
    this->TresholdAutoManual->GetWidgetName());

  this->ThresholdRange->SetParent(threshFrame);
  this->ThresholdRange->Create();
  this->ThresholdRange->SymmetricalInteractionOff();
  this->ThresholdRange->SetCommand(this, "ProcessThresholdCommand");
  this->ThresholdRange->SetStartCommand(this, "ProcessThresholdStartCommand");
  this->Script(
    "pack %s -side left -anchor w -expand n -padx 2 -pady 2", 
    this->ThresholdRange->GetWidgetName());

  this->TresholdApply->SetParent(threshFrame);
  this->TresholdApply->Create();
  this->TresholdApply->SetLabelText("Apply");
  this->Script(
    "pack %s -side top -anchor nw -expand n -padx 2 -pady 2", 
    this->TresholdApply->GetWidgetName());  


  this->ColorTransferFunctionEditor->SetParent(this);
  this->ColorTransferFunctionEditor->Create();
  this->ColorTransferFunctionEditor->ExpandCanvasWidthOff();
  this->ColorTransferFunctionEditor->SetCanvasWidth(300);
  this->ColorTransferFunctionEditor->SetCanvasHeight(150);
  this->ColorTransferFunctionEditor->LabelVisibilityOff ();
  this->ColorTransferFunctionEditor->SetBalloonHelpString(
    "Another color transfer function editor. The point position is now on "
    "top, the point style is an arrow down, guidelines are shown for each "
    "point (useful when combined with an histogram), point indices are "
    "hidden, ticks are displayed in the parameter space, the label "
    "and the parameter range are on top, its width is set explicitly. "
    "The range and histogram are based on a real image data.");
  this->UpdateTransferFunction();
  this->ColorTransferFunctionEditor->SetWholeParameterRangeToFunctionRange();
  this->ColorTransferFunctionEditor->SetVisibleParameterRangeToWholeParameterRange();
  
  this->ColorTransferFunctionEditor->SetPointPositionInValueRangeToTop();
  this->ColorTransferFunctionEditor->SetPointStyleToCursorDown();
  this->ColorTransferFunctionEditor->FunctionLineVisibilityOff();
  this->ColorTransferFunctionEditor->PointGuidelineVisibilityOff();
  // This will disable mobing points on the editor
  // this->ColorTransferFunctionEditor->PointVisibilityOff(); 
  this->ColorTransferFunctionEditor->PointIndexVisibilityOff();
  this->ColorTransferFunctionEditor->SelectedPointIndexVisibilityOff();
  this->ColorTransferFunctionEditor->MidPointEntryVisibilityOff();
  this->ColorTransferFunctionEditor->SharpnessEntryVisibilityOff();
  
  //this->ColorTransferFunctionEditor->SetHistogram(this->Histogram);
  
  this->ColorTransferFunctionEditor->ParameterTicksVisibilityOff();
  this->ColorTransferFunctionEditor->ComputeValueTicksFromHistogramOff();
  
  this->Script(
    "pack %s -side bottom -anchor nw -expand n -padx 2 -pady 20", 
    this->ColorTransferFunctionEditor->GetWidgetName());
  
  
  this->ColorTransferFunctionEditor->SetHistogramStyleToPolyLine();
  this->ColorTransferFunctionEditor->SetHistogramColor(1.0, 0., 0.);
  this->ColorTransferFunctionEditor->SetHistogramPolyLineWidth (5);

  this->ColorTransferFunctionEditor->SetColorRampPositionToCanvas();
  
  this->ColorTransferFunctionEditor->SetColorRampOutlineStyleToNone();
  
  this->ColorTransferFunctionEditor->SetColorRampHeight(100);
    
   this->SetWindowLevel(0, 255);
   this->SetThreshold(0, 255);

  // Override the column sorting behavior by always updating 
}

void vtkKWWindowLevelThresholdEditor::UpdateFromImage()
{
  if (this->ImageData != NULL)
  {   
    this->Histogram->BuildHistogram( this->ImageData->GetPointData()->GetScalars(), 0);
    double *range = this->Histogram->GetRange();
    this->SetWindowLevel(range[0], range[1]);
    this->SetThreshold(range[0], range[1]);

    // avoid crash when Image not set for histogram
    this->ColorTransferFunctionEditor->SetHistogram(this->Histogram);
  }
}
//----------------------------------------------------------------------------
void vtkKWWindowLevelThresholdEditor::UpdateTransferFunction()
{
  this->TransferFunction->RemoveAllPoints();

  double range[2] = {0,255};
  if (this->ImageData)
  {
    this->ImageData->GetScalarRange(range);
  }
  this->TransferFunction->AdjustRange(range);
  this->TransferFunction->SetColorSpaceToRGB();
  this->TransferFunction->AddRGBPoint(range[0], 0, 0, 0);
  //this->TransferFunction->AddRGBPoint(this->GetLowerThreshold(), 179.0/255, 179.0/255, 231.0/255);
  this->TransferFunction->AddRGBPoint(this->GetLowerThreshold(), 0, 0, 0);
  //this->TransferFunction->AddRGBPoint((range[0] + range[1]) * 0.5, 0.0, 1.0, 1.0);
  //this->TransferFunction->AddRGBPoint(this->GetUpperThreshold(), 179.0/255, 179.0/255, 231.0/255);
  this->TransferFunction->AddRGBPoint(this->GetUpperThreshold(), 1, 1, 1);
  this->TransferFunction->AddRGBPoint(range[1], 1, 1, 1);
  this->TransferFunction->SetAlpha(1.0);
  this->TransferFunction->Build();
  this->ColorTransferFunctionEditor->Update();
}



//----------------------------------------------------------------------------
void vtkKWWindowLevelThresholdEditor::SetCommand(vtkObject *object, const char *method)
{
  this->SetObjectMethodCommand(&this->Command, object, method);
}


//----------------------------------------------------------------------------
void vtkKWWindowLevelThresholdEditor::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);

  os << indent << "ColorTransferFunctionEditor: " << this->ColorTransferFunctionEditor << endl;
}


void vtkKWWindowLevelThresholdEditor::ProcessWindowLevelCommand(double min, double max)
{
  double range[2];
  range[0] = min;
  range[1] = max;
  this->UpdateTransferFunction();
  this->InvokeEvent(vtkKWWindowLevelThresholdEditor::ValueChangedEvent, range);
}

void vtkKWWindowLevelThresholdEditor::ProcessWindowLevelStartCommand(double min, double max)
{
  double range[2];
  range[0] = min;
  range[1] = max;
  this->UpdateTransferFunction();
  this->InvokeEvent(vtkKWWindowLevelThresholdEditor::ValueStartChangingEvent, range);
}

void vtkKWWindowLevelThresholdEditor::ProcessThresholdCommand(double min, double max)
{
  double range[2];
  range[0] = min;
  range[1] = max;
  this->UpdateTransferFunction();
  this->InvokeEvent(vtkKWWindowLevelThresholdEditor::ValueChangedEvent, range);
}

void vtkKWWindowLevelThresholdEditor::ProcessThresholdStartCommand(double min, double max)
{
  double range[2];
  range[0] = min;
  range[1] = max;
  this->UpdateTransferFunction();
  this->InvokeEvent(vtkKWWindowLevelThresholdEditor::ValueStartChangingEvent, range);
}
