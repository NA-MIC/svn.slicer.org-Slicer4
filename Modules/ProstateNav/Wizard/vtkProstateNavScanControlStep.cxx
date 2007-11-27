#include "vtkProstateNavScanControlStep.h"

#include "vtkProstateNavGUI.h"
#include "vtkProstateNavLogic.h"

#include "vtkKWFrame.h"
#include "vtkKWFrameWithLabel.h"
#include "vtkKWWizardWidget.h"
#include "vtkKWWizardWorkflow.h"
#include "vtkKWMultiColumnList.h"
#include "vtkKWMultiColumnListWithScrollbars.h"
#include "vtkKWPushButton.h"

#include "vtkSlicerApplication.h"
#include "vtkSlicerFiducialsGUI.h"
#include "vtkSlicerFiducialsLogic.h"
#include "vtkMRMLSelectionNode.h"

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkProstateNavScanControlStep);
vtkCxxRevisionMacro(vtkProstateNavScanControlStep, "$Revision: 1.1 $");

//----------------------------------------------------------------------------
vtkProstateNavScanControlStep::vtkProstateNavScanControlStep()
{
  this->SetName("2/5. Set Scanner Parameters");
  this->SetDescription("Operate the MRI scanner.");

  this->TargetListFrame  = NULL;
  this->MultiColumnList = NULL;
  this->TargetControlFrame = NULL; 
  this->AddButton        = NULL;
  this->RemoveButton     = NULL;
  this->RemoveAllButton  = NULL;

  this->FiducialListNodeID = NULL;
  this->FiducialListNode   = NULL;

}

//----------------------------------------------------------------------------
vtkProstateNavScanControlStep::~vtkProstateNavScanControlStep()
{
  
}

//----------------------------------------------------------------------------
void vtkProstateNavScanControlStep::ShowUserInterface()
{
  this->Superclass::ShowUserInterface();

  vtkKWWizardWidget *wizardWidget = this->GetGUI()->GetWizardWidget();
  vtkKWWidget *parent = wizardWidget->GetClientArea();


  // -----------------------------------------------------------------
  // Target List Frame

  if (!this->TargetListFrame)
    {
    this->TargetListFrame = vtkKWFrame::New();
    this->TargetListFrame->SetParent(parent);
    this->TargetListFrame->Create();
    }

  this->Script("pack %s -side top -anchor nw -expand n -padx 2 -pady 2",
                this->TargetListFrame->GetWidgetName());

  if (!this->MultiColumnList)
    {
    // add the multicolumn list to show the points
    this->MultiColumnList = vtkKWMultiColumnListWithScrollbars::New();
    this->MultiColumnList->SetParent(TargetListFrame);
    this->MultiColumnList->Create();
    this->MultiColumnList->SetHeight(4);
    this->MultiColumnList->GetWidget()->SetSelectionTypeToCell();
    this->MultiColumnList->GetWidget()->MovableRowsOff();
    this->MultiColumnList->GetWidget()->MovableColumnsOff();

    const char* labels[] =
      { "Name", "X", "Y", "Z", "OrW", "OrX", "OrY", "OrZ" };
    const int widths[] = 
      { 6, 6, 6, 6, 6, 6, 6, 6 };

    for (int col = 0; col < 8; col ++)
      {
      this->MultiColumnList->GetWidget()->AddColumn(labels[col]);
      this->MultiColumnList->GetWidget()->SetColumnWidth(col, widths[col]);
      this->MultiColumnList->GetWidget()->SetColumnAlignmentToLeft(col);
      this->MultiColumnList->GetWidget()->ColumnEditableOff(col);
      }
    }

  this->Script( "pack %s -side top -anchor nw -expand n -padx 2 -pady 2",
                this->MultiColumnList->GetWidgetName());


  // -----------------------------------------------------------------
  // Target Control Frame

  if (!this->TargetControlFrame)
    {
    this->TargetControlFrame = vtkKWFrame::New();
    this->TargetControlFrame->SetParent(TargetListFrame);
    this->TargetControlFrame->Create();
    }

  this->Script("pack %s -side top -anchor nw -expand n -padx 2 -pady 2",
               this->TargetControlFrame->GetWidgetName());

  if (!this->AddButton)
    {
    this->AddButton = vtkKWPushButton::New();
    this->AddButton->SetParent (this->TargetControlFrame);
    this->AddButton->Create();
    this->AddButton->SetText("Add Target");
    this->AddButton->SetBalloonHelpString("Move the robot to the position");
    this->AddButton->AddObserver(vtkKWPushButton::InvokedEvent,
                                  (vtkCommand *)this->GUICallbackCommand);
    }

  if (!this->RemoveButton)
    {
    this->RemoveButton = vtkKWPushButton::New();
    this->RemoveButton->SetParent (this->TargetControlFrame);
    this->RemoveButton->Create();
    this->RemoveButton->SetText("Remove Target");
    this->RemoveButton->SetBalloonHelpString("Move the robot to the position");
    this->RemoveButton->AddObserver(vtkKWPushButton::InvokedEvent,
                                  (vtkCommand *)this->GUICallbackCommand);
    }

  if (!this->RemoveAllButton)
    {
    this->RemoveAllButton = vtkKWPushButton::New();
    this->RemoveAllButton->SetParent (this->TargetControlFrame);
    this->RemoveAllButton->Create();
    this->RemoveAllButton->SetText("Remove All");
    this->RemoveAllButton->SetBalloonHelpString("Move the robot to the position");
    this->RemoveAllButton->AddObserver(vtkKWPushButton::InvokedEvent,
                                  (vtkCommand *)this->GUICallbackCommand);
    }

  this->Script("pack %s %s %s -side left -anchor nw -expand n -padx 2 -pady 2",
               this->AddButton->GetWidgetName(),
               this->RemoveButton->GetWidgetName(),
               this->RemoveAllButton->GetWidgetName());


  // -----------------------------------------------------------------
  // MRML Event Observer

  if (!this->FiducialListNodeID)
    {
    vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();

    // Get a pointer to the Fiducials module
    vtkSlicerFiducialsGUI *fidGUI
      = (vtkSlicerFiducialsGUI*)app->GetModuleGUIByName("Fiducials");
    fidGUI->Enter();

    // Create New Fiducial list for Prostate Module
    vtkSlicerFiducialsLogic *fidLogic = (vtkSlicerFiducialsLogic*)(fidGUI->GetLogic());
    vtkMRMLFiducialListNode *newList = fidLogic->AddFiducialList();

    if (newList != NULL)
      {
      // Change the name of the list
      newList->SetName(this->MRMLScene->GetUniqueNameByString("PM"));

      fidGUI->SetFiducialListNodeID(newList->GetID());
      newList->Delete();
      }
    else
      {
        vtkErrorMacro("Unable to add a new fid list via the logic\n");
      }
    // now get the newly active node 
    this->FiducialListNodeID = fidGUI->GetFiducialListNodeID();
    this->FiducialListNode 
      = (vtkMRMLFiducialListNode *)this->MRMLScene->GetNodeByID(this->FiducialListNodeID);
    if (this->FiducialListNode == NULL)
      {
      vtkErrorMacro ("ERROR adding a new fiducial list for the point...\n");
      return;
      }
  
    this->MRMLScene->SaveStateForUndo(this->FiducialListNode);

    vtkMRMLSelectionNode *selnode;
    if (this->GetGUI()->GetApplicationLogic())
      {
      selnode = this->GetGUI()->GetApplicationLogic()->GetSelectionNode();
      }
    this->UpdateMRMLObserver(selnode);
    }

}

//----------------------------------------------------------------------------
void vtkProstateNavScanControlStep::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}

//----------------------------------------------------------------------------
void vtkProstateNavScanControlStep::ProcessGUIEvents(vtkObject *caller,
                                         unsigned long event, void *callData)
{

  std::cerr << "vtkProstateNavScanControlStep::ProcessGUIEvents()" << std::endl;

  vtkMRMLFiducialListNode *activeFiducialListNode 
    = (vtkMRMLFiducialListNode *)this->MRMLScene->GetNodeByID(this->FiducialListNodeID);
  

  // -----------------------------------------------------------------
  // Add Button Pressed

  if (this->AddButton == vtkKWPushButton::SafeDownCast(caller)
      && event == vtkKWPushButton::InvokedEvent)
    {
    this->MRMLScene->SaveStateForUndo();
    int modelIndex = this->FiducialListNode->AddFiducial();
    if ( modelIndex < 0 ) 
      {
      vtkErrorMacro ("ERROR adding a new fiducial point\n");
      }
    }

  // -----------------------------------------------------------------
  // Remove Button Pressed

  else if (this->RemoveButton == vtkKWPushButton::SafeDownCast(caller)
      && event == vtkKWPushButton::InvokedEvent)
    {
    const char * confirmDelete 
      = ((vtkSlicerApplication *)this->GetApplication())->GetConfirmDelete();
    int confirmDeleteFlag = 0;
    if (confirmDelete != NULL &&
        strncmp(confirmDelete, "1", 1) == 0)
      {
      confirmDeleteFlag = 1;
      }
    this->MRMLScene->SaveStateForUndo();    // save state for undo

    // get the row that was last selected

    int numRows = this->MultiColumnList->GetWidget()->GetNumberOfSelectedRows();
    //
    //  should be implemented
    //
    if (numRows == 1)
      {
        int row[1];
        this->MultiColumnList->GetWidget()->GetSelectedRows(row);
        if (confirmDeleteFlag)
          {
          // confirm that really want to remove this fiducial
          std::cout << "Removing fiducial " << row[0] << endl;
          this->FiducialListNode->RemoveFiducial(row[0]);
          }
      }
    }


  // -----------------------------------------------------------------
  // Remove All Button Pressed

  else if (this->AddButton == vtkKWPushButton::SafeDownCast(caller)
           && event == vtkKWPushButton::InvokedEvent)
    {
    this->MRMLScene->SaveStateForUndo();
    this->FiducialListNode->RemoveAllFiducials();
    }
  
}


//----------------------------------------------------------------------------
void vtkProstateNavScanControlStep::ProcessMRMLEvents(vtkObject *caller,
                                         unsigned long event, void *callData)
{
  std::cerr << "vtkProstateNavScanControlStep::ProcessMRMLEvents() is called. " << std::cerr;

  vtkMRMLSelectionNode *selnode;
  if (this->GetGUI()->GetApplicationLogic())
    {
    selnode = this->GetGUI()->GetApplicationLogic()->GetSelectionNode();
    }


  // -----------------------------------------------------------------
  // Update MRML Observer

  if (selnode != NULL 
      && vtkMRMLSelectionNode::SafeDownCast(caller) == selnode
      && event == vtkCommand::ModifiedEvent)
    {

    if (selnode->GetActiveFiducialListID() != NULL &&
        this->FiducialListNodeID != NULL)
      {
      if (strcmp(selnode->GetActiveFiducialListID(), this->FiducialListNodeID) != 0)
        {
        if (!selnode->GetActiveFiducialListID())
          {
          this->UpdateMRMLObserver(selnode);
          }
        }
      }
    }

  // -----------------------------------------------------------------
  // Fiducial Modified

  if (event == vtkCommand::WidgetValueChangedEvent)
    {
    vtkDebugMacro("got a widget value changed event... the list node was changed.\n");
    }

  vtkMRMLFiducialListNode *node = vtkMRMLFiducialListNode::SafeDownCast(caller);
  if (!node)
    {
    return;
    }
  /*
  vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
  vtkSlicerFiducialsGUI *fidGUI
    = (vtkSlicerFiducialsGUI*)app->GetModuleGUIByName("Fiducials");
  vtkMRMLFiducialListNode *activeFiducialListNode 
    = (vtkMRMLFiducialListNode *)this->MRMLScene->GetNodeByID(fidGUI->GetFiducialListNodeID());
  */
  
  // -----------------------------------------------------------------
  // Modified Event

  if (node == this->FiducialListNode && event == vtkCommand::ModifiedEvent)
    {
    SetGUIFromList(this->FiducialListNode);
    std::cerr << "vtkProstateNavScanControlStep: ModifiedEvent!" << std::endl;
    }

  // -----------------------------------------------------------------
  // Fiducial Modified Event

  else if (node == this->FiducialListNode && event == vtkMRMLFiducialListNode::FiducialModifiedEvent)
    {
    // Update table here !!!
    SetGUIFromList(this->FiducialListNode);
    std::cerr << "vtkProstateNavScanControlStep: FiducialModifiedEvent!" << std::endl;
    }
  
  // -----------------------------------------------------------------
  // Display Modified Event

  else if (node == this->FiducialListNode && event == vtkMRMLFiducialListNode::DisplayModifiedEvent)
    {
    SetGUIFromList(this->FiducialListNode);
    std::cerr << "vtkProstateNavScanControlStep: DisplayModifiedEvent!" << std::endl;
    }

}

//----------------------------------------------------------------------------
void vtkProstateNavScanControlStep::UpdateMRMLObserver(vtkMRMLSelectionNode* selnode)
{

  std::cerr << "vtkProstateNavScanControlStep::UpdateMRMLObserver()" << std::endl;

  vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
  vtkSlicerFiducialsGUI *fidGUI
    = (vtkSlicerFiducialsGUI*) app->GetModuleGUIByName("Fiducials");

  vtkMRMLFiducialListNode* fidlist = 
    vtkMRMLFiducialListNode::SafeDownCast(this->MRMLScene
                                          ->GetNodeByID(this->FiducialListNodeID));
  if (selnode != NULL)
    {
    std::cerr << "selnode != 0;" << std::endl;
    // is the active fid list id out of synch with our selection?
    vtkIntArray *events = vtkIntArray::New();
    events->InsertNextValue(vtkCommand::ModifiedEvent);
    events->InsertNextValue(vtkMRMLFiducialListNode::DisplayModifiedEvent);
    events->InsertNextValue(vtkMRMLFiducialListNode::FiducialModifiedEvent);

    /*
    vtkMRMLFiducialListNode *activeFiducialListNode 
      = (vtkMRMLFiducialListNode *)this->MRMLScene
      ->GetNodeByID(this->FiducialListNodeID);
    */
    
    vtkObject *oldNode = (fidlist);
    this->MRMLObserverManager
      ->SetAndObserveObjectEvents(vtkObjectPointer(&(this->FiducialListNode)),(fidlist),(events));
    if ( oldNode != (fidlist) )
      {
      this->InvokeEvent (vtkCommand::ModifiedEvent);
      } 
    }
}

//----------------------------------------------------------------------------
void vtkProstateNavScanControlStep::SetGUIFromList(vtkMRMLFiducialListNode * activeFiducialListNode)
{

  if (activeFiducialListNode == NULL)    //clear out the list box
    {
    if (this->MultiColumnList)
      {
      if (this->MultiColumnList->GetWidget()->GetNumberOfRows() != 0)
        {
        this->MultiColumnList->GetWidget()->DeleteAllRows();
        }
      }
    return;
    }
    
  int numPoints = activeFiducialListNode->GetNumberOfFiducials();
  bool deleteFlag = true;

  if (numPoints != this->MultiColumnList->GetWidget()->GetNumberOfRows())
    {
    // clear out the multi column list box and fill it in with the
    // new list
    this->MultiColumnList->GetWidget()->DeleteAllRows();
    }
  else
    {
    deleteFlag = false;
    }
        
  float *xyz;
  float *wxyz;

  for (int row = 0; row < numPoints; row++)
    {
    if (deleteFlag)
      {
      // add a row for this point
      this->MultiColumnList->GetWidget()->AddRow();
      }

    xyz = activeFiducialListNode->GetNthFiducialXYZ(row);
    wxyz = activeFiducialListNode->GetNthFiducialOrientation(row);

    if (xyz == NULL)
      {
      vtkErrorMacro ("SetGUIFromList: ERROR: got null xyz for point " << row << endl);
      }

    if (activeFiducialListNode->GetNthFiducialLabelText(row) != NULL)
      {
      if (strcmp(this->MultiColumnList->GetWidget()->GetCellText(row,0),
                 activeFiducialListNode->GetNthFiducialLabelText(row)) != 0)
        {
        this->MultiColumnList->GetWidget()
          ->SetCellText(row,0,activeFiducialListNode->GetNthFiducialLabelText(row));
        }               
      }
    else
      {
      if (strcmp(this->MultiColumnList->GetWidget()->GetCellText(row, 0), "(none)") != 0)
        {
        this->MultiColumnList->GetWidget()->SetCellText(row,0,"(none)");
        }
      }

    // selected
    /*
    if (deleteFlag ||
        this->MultiColumnList->GetWidget()->GetCellTextAsInt(row,this->SelectedColumn)
        != (activeFiducialListNode->GetNthFiducialSelected(row) ? 1 : 0))
      {
      this->MultiColumnList->GetWidget()
        ->SetCellTextAsInt(row,this->SelectedColumn,
                           (activeFiducialListNode->GetNthFiducialSelected(row) ? 1 : 0));
      this->MultiColumnList->GetWidget()->SetCellWindowCommandToCheckButton(row,this->SelectedColumn);
      }
    */
    vtkKWMultiColumnList* columnList = this->MultiColumnList->GetWidget();
    if (xyz != NULL)
      {
      for (int i = 0; i < 3; i ++) // for position (x, y, z)
        {
        if (deleteFlag || columnList->GetCellTextAsDouble(row,1+i) != xyz[i])
          {
          columnList->SetCellTextAsDouble(row,1+i,xyz[i]);
          }
        }
      }
    if (wxyz != NULL)
      {
      for (int i = 0; i < 4; i ++) // for orientation (w, x, y, z)
        {
        if (deleteFlag || columnList->GetCellTextAsDouble(row, 4+i) != wxyz[i])
          {
          columnList->SetCellTextAsDouble(row,4+i,wxyz[i]);
          }
        }
      }
    }

  vtkDebugMacro("Now going to update GUI from the logic's active list");

}
