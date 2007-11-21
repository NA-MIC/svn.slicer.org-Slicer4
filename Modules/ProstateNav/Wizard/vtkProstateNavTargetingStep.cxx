#include "vtkProstateNavTargetingStep.h"

#include "vtkProstateNavGUI.h"
#include "vtkProstateNavLogic.h"

#include "vtkKWFrame.h"
#include "vtkKWFrameWithLabel.h"
#include "vtkKWWizardWidget.h"
#include "vtkKWWizardWorkflow.h"
#include "vtkKWMultiColumnList.h"
#include "vtkKWMultiColumnListWithScrollbars.h"
#include "vtkKWLabel.h"
#include "vtkKWMatrixWidget.h"
#include "vtkKWMatrixWidgetWithLabel.h"
#include "vtkKWPushButton.h"

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkProstateNavTargetingStep);
vtkCxxRevisionMacro(vtkProstateNavTargetingStep, "$Revision: 1.1 $");

//----------------------------------------------------------------------------
vtkProstateNavTargetingStep::vtkProstateNavTargetingStep()
{
  this->SetName("4/5. Targeting");
  this->SetDescription("Set target points.");

  this->MainFrame          = NULL;
  this->TargetListFrame    = NULL;
  this->TargetControlFrame = NULL;

  this->TargetListColumnList = NULL;
  this->NeedlePositionMatrix = NULL;
  this->NeedleNormalMatrix   = NULL;

  this->MoveButton = NULL;
  this->StopButton = NULL;

}
//----------------------------------------------------------------------------
vtkProstateNavTargetingStep::~vtkProstateNavTargetingStep()
{
}

//----------------------------------------------------------------------------
void vtkProstateNavTargetingStep::ShowUserInterface()
{
  this->Superclass::ShowUserInterface();

  vtkKWWizardWidget *wizardWidget = this->GetGUI()->GetWizardWidget();
  wizardWidget->GetCancelButton()->SetEnabled(0);

  vtkKWWidget *parent = wizardWidget->GetClientArea();

  if (!this->MainFrame)
    {
    this->MainFrame = vtkKWFrame::New();
    this->MainFrame->SetParent(parent);
    this->MainFrame->Create();
    }

  this->Script( "pack %s -side top -anchor nw -expand n -padx 2 -pady 2",
                this->MainFrame->GetWidgetName());
    
  if (!this->TargetListFrame)
    {
    this->TargetListFrame = vtkKWFrame::New();
    this->TargetListFrame->SetParent(this->MainFrame);
    this->TargetListFrame->Create();
    }

  this->Script( "pack %s -side top -anchor nw -expand n -padx 2 -pady 2",
                this->TargetListFrame->GetWidgetName());
  
  if (!this->TargetListColumnList)
    {
    this->TargetListColumnList = vtkKWMultiColumnListWithScrollbars::New();
    this->TargetListColumnList->SetParent(this->TargetListFrame);
    this->TargetListColumnList->Create();
    this->TargetListColumnList->SetHeight(1);
    this->TargetListColumnList->GetWidget()->SetSelectionTypeToRow();
    this->TargetListColumnList->GetWidget()->MovableRowsOff();
    this->TargetListColumnList->GetWidget()->MovableColumnsOff();

    const char* labels[] =
      {
      "Id",
      "Tip Pos. (R, A, S)",
      "Norm. (NR, NA, NS)"
      };
    const int widths[] = {7, 18, 18};

    for (int col = 0; col < 3; col ++)
      {
      this->TargetListColumnList->GetWidget()->AddColumn(labels[col]);
      this->TargetListColumnList->GetWidget()->SetColumnWidth(col, widths[col]);
      this->TargetListColumnList->GetWidget()->SetColumnAlignmentToLeft(col);
      this->TargetListColumnList->GetWidget()->ColumnEditableOff(col);
      }
    }

  this->Script( "pack %s -side top -anchor nw -expand n -padx 2 -pady 2",
                this->TargetListColumnList->GetWidgetName());
  
  if (!this->NeedlePositionMatrix)
    {
    this->NeedlePositionMatrix = vtkKWMatrixWidgetWithLabel::New();
    this->NeedlePositionMatrix->SetParent(this->TargetListFrame);
    this->NeedlePositionMatrix->Create();
    this->NeedlePositionMatrix->SetLabelText("Position (R, A, S):");
    this->NeedlePositionMatrix->ExpandWidgetOff();
    this->NeedlePositionMatrix->GetLabel()->SetWidth(18);
    this->NeedlePositionMatrix->SetBalloonHelpString("Set the needle position");

    vtkKWMatrixWidget *matrix =  this->NeedlePositionMatrix->GetWidget();
    matrix->SetNumberOfColumns(3);
    matrix->SetNumberOfRows(1);
    matrix->SetElementWidth(12);
    matrix->SetElementChangedCommandTriggerToAnyChange();
    }

  if (!this->NeedleNormalMatrix)
    {
    this->NeedleNormalMatrix = vtkKWMatrixWidgetWithLabel::New();
    this->NeedleNormalMatrix->SetParent(this->TargetListFrame);
    this->NeedleNormalMatrix->Create();
    this->NeedleNormalMatrix->SetLabelText("Normal (NR, NA, NS):");
    this->NeedleNormalMatrix->ExpandWidgetOff();
    this->NeedleNormalMatrix->GetLabel()->SetWidth(18);
    this->NeedleNormalMatrix->SetBalloonHelpString("Set the needle orientation");

    vtkKWMatrixWidget *matrix =  this->NeedleNormalMatrix->GetWidget();
    matrix->SetNumberOfColumns(3);
    matrix->SetNumberOfRows(1);
    matrix->SetElementWidth(12);
    matrix->SetElementChangedCommandTriggerToAnyChange();
    }

  this->Script("pack %s %s -side top -anchor nw -expand n -padx 2 -pady 2",
               this->NeedlePositionMatrix->GetWidgetName(),
               this->NeedleNormalMatrix->GetWidgetName());

  if (!this->TargetControlFrame)
    {
    this->TargetControlFrame = vtkKWFrame::New();
    this->TargetControlFrame->SetParent(this->MainFrame);
    this->TargetControlFrame->Create();
    }

  this->Script("pack %s -side top -anchor nw -expand n -padx 2 -pady 2",
               this->TargetControlFrame->GetWidgetName());
  
  if (!this->MoveButton)
    {
    this->MoveButton = vtkKWPushButton::New();
    this->MoveButton->SetParent (this->TargetControlFrame);
    this->MoveButton->Create();
    this->MoveButton->SetText("Move");
    this->MoveButton->SetBalloonHelpString("Move the robot to the position");
    this->MoveButton->AddObserver(vtkKWPushButton::InvokedEvent,
                                  (vtkCommand *)this->GUICallbackCommand);
    }

  if (!this->StopButton)
    {
    this->StopButton = vtkKWPushButton::New();
    this->StopButton->SetParent (this->TargetControlFrame);
    this->StopButton->Create();
    this->StopButton->SetText("Stop");
    this->StopButton->SetBalloonHelpString("Stop the robot");
    this->StopButton->AddObserver(vtkKWPushButton::InvokedEvent,
                                  (vtkCommand *)this->GUICallbackCommand);
    }

  this->Script("pack %s %s -side left -anchor nw -expand n -padx 2 -pady 2",
               this->MoveButton->GetWidgetName(),
               this->StopButton->GetWidgetName());

      
}

//----------------------------------------------------------------------------
void vtkProstateNavTargetingStep::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}

//----------------------------------------------------------------------------
void vtkProstateNavTargetingStep::ProcessGUIEvents(vtkObject *caller,
                                          unsigned long event, void *callData)
{

  if (this->MoveButton == vtkKWPushButton::SafeDownCast(caller)
      && event == vtkKWPushButton::InvokedEvent)
    {
    if (this->Logic && this->NeedlePositionMatrix && this->NeedleNormalMatrix)
      {
      float pr, pa, ps, nr, na, ns, tr, ta, ts;

      vtkKWMatrixWidget* matrix = this->NeedlePositionMatrix->GetWidget();
      pr = (float) matrix->GetElementValueAsDouble(0, 0);
      pa = (float) matrix->GetElementValueAsDouble(0, 1);
      ps = (float) matrix->GetElementValueAsDouble(0, 2);

      matrix = this->NeedleNormalMatrix->GetWidget();
      nr = (float) matrix->GetElementValueAsDouble(0, 0);
      na = (float) matrix->GetElementValueAsDouble(0, 1);
      ns = (float) matrix->GetElementValueAsDouble(0, 2);

      // J. Tokuda 11/21/2007: 
      // Calculate trans-normal vector of the needle by assuming z-element is 0.
      // This is a temporal implementation.
      tr = na;
      ta = -nr;
      ts = 0.0;

      this->Logic->RobotMoveTo(pr, pa, ps, nr, na, ns, tr, ta, ts);

      }
    }
  else if (this->StopButton == vtkKWPushButton::SafeDownCast(caller)
      && event == vtkKWPushButton::InvokedEvent)
    {
    }           

}

