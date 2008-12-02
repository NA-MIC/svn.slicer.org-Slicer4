#include "vtkObject.h"
#include "vtkObjectFactory.h"
#include "vtkCommand.h"
#include "vtkKWWidget.h"
#include "vtkSlicerModelsGUI.h"
#include "vtkSlicerApplication.h"
#include "vtkSlicerModuleLogic.h"
#include "vtkSlicerVisibilityIcons.h"
#include "vtkSlicerModuleCollapsibleFrame.h"

#include "vtkKWMessage.h"
#include "vtkKWMultiColumnList.h"
#include "vtkKWMultiColumnListWithScrollbars.h"

#include "vtkKWFrameWithLabel.h"

#include "vtkKWMenu.h"
#include "vtkKWMenuButton.h"
#include "vtkKWMenuButtonWithLabel.h"

#include "vtkSlicerROIGUI.h"
#include "vtkSlicerROIDisplayWidget.h"

//---------------------------------------------------------------------------
vtkStandardNewMacro (vtkSlicerROIGUI );
vtkCxxRevisionMacro ( vtkSlicerROIGUI, "$Revision: 1.0 $");


//---------------------------------------------------------------------------
vtkSlicerROIGUI::vtkSlicerROIGUI ( )
{
  this->Logic = NULL;
  this->ROIListNodeID = NULL;
  //this->VolumeNodeID = NULL;
  this->ROIListNode = NULL;
  this->ROINode = NULL;
  this->ROISelectorWidget = NULL;
  this->ROIListSelectorWidget = NULL;
  this->VolumeNodeSelectorWidget = NULL;
  this->AddROIButton = NULL;
  this->RemoveROIButton = NULL;
  this->RemoveROIListButton= NULL;
  this->VisibilityToggle = NULL;
  this->VisibilityIcons = NULL;
  this->ROIColorButton = NULL;
  this->ROISelectedColorButton = NULL;
  this->ROITextScale = NULL;
  this->ROIOpacity = NULL;
  this->MultiColumnList = NULL;
  this->NumberOfColumns = 8;


  this->ROIDisplayWidget = NULL;
}

//---------------------------------------------------------------------------
vtkSlicerROIGUI::~vtkSlicerROIGUI ( )
{
  this->SetModuleLogic (NULL);
  this->SetROIListNodeID(NULL);
  
  /*if (this->VolumeNodeID)
    {
    delete [] this->VolumeNodeID;
    this->VolumeNodeID = NULL;
    }*/
  vtkSetMRMLNodeMacro(this->ROINode, NULL);
  vtkSetMRMLNodeMacro(this->ROIListNode, NULL);

  if (this->ROIListSelectorWidget)
    {
    this->ROIListSelectorWidget->SetParent(NULL);
    this->ROIListSelectorWidget->Delete();
    this->ROIListSelectorWidget = NULL;
    }
  if (this->ROISelectorWidget)
    {
    this->ROISelectorWidget->SetParent(NULL);
    this->ROISelectorWidget->Delete();
    this->ROISelectorWidget = NULL;
    }
  if ( this->VolumeNodeSelectorWidget )
    {
    this->VolumeNodeSelectorWidget->SetParent(NULL);
    this->VolumeNodeSelectorWidget->Delete();
    this->VolumeNodeSelectorWidget = NULL;
    }
  if ( this->AddROIButton )
    {
    this->AddROIButton->SetParent ( NULL );
    this->AddROIButton->Delete();
    this->AddROIButton = NULL;
    }
  if ( this->RemoveROIButton )
    {
    this->RemoveROIButton->SetParent ( NULL );
    this->RemoveROIButton->Delete();
    this->RemoveROIButton = NULL;
    }
  if ( this->RemoveROIListButton )
    {
    this->RemoveROIListButton->SetParent ( NULL );
    this->RemoveROIListButton->Delete();
    this->RemoveROIListButton = NULL;
    }
  if ( this->VisibilityToggle )
    {
    this->VisibilityToggle->SetParent ( NULL );
    this->VisibilityToggle->Delete();
    this->VisibilityToggle = NULL;
    }
  if ( this->VisibilityIcons ) {
    this->VisibilityIcons->Delete  ( );
    this->VisibilityIcons = NULL;
    }
  if ( this->ROIColorButton ) {
    this->ROIColorButton->SetParent ( NULL );
    this->ROIColorButton->Delete  ( );
    this->ROIColorButton = NULL;
    }
  if ( this->ROISelectedColorButton ) {
    this->ROISelectedColorButton->SetParent ( NULL );
    this->ROISelectedColorButton->Delete  ( );
    this->ROISelectedColorButton = NULL;
    }
  if ( this->ROITextScale ) {
    this->ROITextScale->SetParent ( NULL );
    this->ROITextScale->Delete  ( );
    this->ROITextScale = NULL;
    }
  if ( this->MultiColumnList ) {
    this->MultiColumnList->SetParent ( NULL );
    this->MultiColumnList->Delete  ( );
    this->MultiColumnList = NULL;
    }
  if ( this->ROIOpacity ) {
    this->ROIOpacity->SetParent ( NULL );
    this->ROIOpacity->Delete  ( );
    this->ROIOpacity = NULL;
    }
  if (this->ROIDisplayWidget) {
    this->ROIDisplayWidget->SetParent(NULL);
    this->ROIDisplayWidget->Delete();
    this->ROIDisplayWidget = NULL;
    }
 
  return;
}

//---------------------------------------------------------------------------
void vtkSlicerROIGUI::PrintSelf ( ostream& os, vtkIndent indent )
{
  this->vtkObject::PrintSelf ( os, indent );

  os << indent << "SlicerROIGUI: " << this->GetClassName ( ) << "\n";
  os << indent << "Logic: " << this->GetLogic ( ) << "\n";
  }

//---------------------------------------------------------------------------
void vtkSlicerROIGUI::RemoveGUIObservers ( )
{
  this->MultiColumnList->GetWidget()->RemoveObservers(vtkKWMultiColumnList::SelectionChangedEvent, (vtkCommand *)this->GUICallbackCommand);

  this->ROISelectorWidget->RemoveObservers(vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand);

  this->ROIListSelectorWidget->RemoveObservers(vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand);

  this->VolumeNodeSelectorWidget->RemoveObservers(vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand);

  this->AddROIButton->RemoveObservers ( vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );

  this->RemoveROIButton->RemoveObservers ( vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );

  this->RemoveROIListButton->RemoveObservers ( vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );

  this->VisibilityToggle->RemoveObservers (vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );

  this->ROIColorButton->RemoveObservers (vtkKWChangeColorButton::ColorChangedEvent, (vtkCommand *)this->GUICallbackCommand);

  this->ROISelectedColorButton->RemoveObservers (vtkKWChangeColorButton::ColorChangedEvent, (vtkCommand *)this->GUICallbackCommand);

  this->ROITextScale->RemoveObservers (vtkKWScale::ScaleValueChangedEvent, (vtkCommand *)this->GUICallbackCommand);

  this->ROIOpacity->RemoveObservers (vtkKWScale::ScaleValueChangedEvent, (vtkCommand *)this->GUICallbackCommand);

  this->MRMLScene->RemoveObservers(vtkMRMLScene::NodeRemovedEvent, (vtkCommand *)this->GUICallbackCommand);
}

//---------------------------------------------------------------------------
void vtkSlicerROIGUI::AddGUIObservers ( )
{
  this->MultiColumnList->GetWidget()->AddObserver(vtkKWMultiColumnList::SelectionChangedEvent, (vtkCommand *)this->GUICallbackCommand);

  this->ROISelectorWidget->AddObserver(vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand );

  this->ROIListSelectorWidget->AddObserver(vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand );

  this->VolumeNodeSelectorWidget->AddObserver(vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand );

  this->AddROIButton->AddObserver ( vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );

  this->RemoveROIButton->AddObserver ( vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );

  this->RemoveROIListButton->AddObserver ( vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );

  this->VisibilityToggle->AddObserver (vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );

  this->ROIColorButton->AddObserver (vtkKWChangeColorButton::ColorChangedEvent, (vtkCommand *)this->GUICallbackCommand );

  this->ROISelectedColorButton->AddObserver (vtkKWChangeColorButton::ColorChangedEvent, (vtkCommand *)this->GUICallbackCommand );

  this->ROITextScale->AddObserver (vtkKWScale::ScaleValueChangedEvent, (vtkCommand *)this->GUICallbackCommand);

  this->ROIOpacity->AddObserver (vtkKWScale::ScaleValueChangedEvent, (vtkCommand *)this->GUICallbackCommand);

  // observe the scene for node deleted events
  if (this->MRMLScene)
    {
    if (this->MRMLScene->HasObserver(vtkMRMLScene::NodeRemovedEvent, (vtkCommand *)this->GUICallbackCommand) != 1)
      {
      this->MRMLScene->AddObserver(vtkMRMLScene::NodeRemovedEvent, (vtkCommand *)this->GUICallbackCommand);
      }
    else
      {
      vtkDebugMacro("MRML scene already has the node removed event being watched by the roi gui");
      }
    }
}


//---------------------------------------------------------------------------
void vtkSlicerROIGUI::ProcessGUIEvents ( vtkObject *caller,
                                        unsigned long event, void *callData )
  {
  // process ROI list node selector events
  vtkSlicerNodeSelectorWidget *roiSelector = 
    vtkSlicerNodeSelectorWidget::SafeDownCast(caller);
  if (roiSelector == this->ROISelectorWidget &&
    event == vtkSlicerNodeSelectorWidget::NodeSelectedEvent )
    {
    vtkMRMLROINode *roi =vtkMRMLROINode::SafeDownCast(this->ROISelectorWidget->GetSelected());
    if (roi!= NULL)
      {
      this->ROIDisplayWidget->SetROINode(roi);
      }
    return;
    }
  vtkSlicerNodeSelectorWidget *ROIListSelector = 
    vtkSlicerNodeSelectorWidget::SafeDownCast(caller);
  if (ROIListSelector == this->ROIListSelectorWidget &&
    event == vtkSlicerNodeSelectorWidget::NodeSelectedEvent )
    {
    vtkDebugMacro("vtkSlicerROIGUI: ProcessGUIEvent Node Selector Event: " << event << ".\n");
    vtkMRMLROIListNode *ROIList =
      vtkMRMLROIListNode::SafeDownCast(this->ROIListSelectorWidget->GetSelected());
    if (ROIList != NULL)
      {
      this->SetROIListNode(ROIList);
      }
    return;
    }

  vtkMRMLROIListNode *ROIList =
    vtkMRMLROIListNode::SafeDownCast(caller);
  if (ROIList == this->MRMLScene->GetNodeByID(this->GetROIListNodeID()) &&
    event == vtkCommand::ModifiedEvent)
    {
    return;
    }

  vtkMRMLROIListNode *activeROIListNode = (vtkMRMLROIListNode *)this->MRMLScene->GetNodeByID(this->GetROIListNodeID());

  if (activeROIListNode == NULL)
    {
    vtkErrorMacro ("ERROR: No ROI List, adding one first!\n");
    vtkMRMLROIListNode *newList = this->GetLogic()->AddROIList();      
    if (newList != NULL)
      {
      this->SetROIListNodeID(newList->GetID());
      newList->Delete();
      }
    else
      {
      vtkErrorMacro("Unable to add a new ROI list via the logic\n");
      }
    // now get the newly active node 
    activeROIListNode = (vtkMRMLROIListNode *)this->MRMLScene->GetNodeByID(this->GetROIListNodeID());
    if (activeROIListNode == NULL)
      {
      vtkErrorMacro ("ERROR adding a new ROI list for the point...\n");
      return;
      }
    }
  // save state for undo
  //this->MRMLScene->SaveStateForUndo(activeROIListNode);

  vtkSlicerNodeSelectorWidget *VlolumeSelector = 
    vtkSlicerNodeSelectorWidget::SafeDownCast(caller);
  if (VlolumeSelector == this->VolumeNodeSelectorWidget &&
    event == vtkSlicerNodeSelectorWidget::NodeSelectedEvent )
    {
    vtkDebugMacro("vtkSlicerROIGUI: ProcessGUIEvent volume Node Selector Event: " << event << ".\n");
    vtkMRMLVolumeNode *VolumeNode =
      vtkMRMLVolumeNode::SafeDownCast(this->VolumeNodeSelectorWidget->GetSelected());
    if (VolumeNode == NULL)
      {
      // this->SetVolumeNodeID(NULL);
      activeROIListNode->SetVolumeNodeID(NULL);
      activeROIListNode->SetAllVolumeNodeID();
      }
    else
      {
      //this->SetVolumeNodeID(VolumeNode->GetID());
      activeROIListNode->SetVolumeNodeID(VolumeNode->GetID());
      activeROIListNode->SetAllVolumeNodeID();
      activeROIListNode->UpdateIJK();
      }
    }

  char* ActiveVolumeNodeID = activeROIListNode->GetVolumeNodeID();

  vtkKWPushButton *button = vtkKWPushButton::SafeDownCast(caller);
  if (button == this->AddROIButton  && event ==  vtkKWPushButton::InvokedEvent)
    {
    vtkDebugMacro("vtkSlicerROIGUI: ProcessGUIEvent: Add ROI Button event: " << event << ".\n");
    // save state for undo
    //this->MRMLScene->SaveStateForUndo();

    // add a ROI, get the index of the new ROI
    int modelIndex = activeROIListNode->AddROI();
    if ( modelIndex < 0 ) 
      {
      // TODO: generate an error...
      vtkErrorMacro ("ERROR adding a new ROI\n");
      return;
      }
    vtkMRMLROINode *roi = activeROIListNode->GetNthROINode(modelIndex);
    this->ROIDisplayWidget->SetROINode(roi);
    }

  if (button == this->RemoveROIButton && event == vtkKWPushButton::InvokedEvent)
    {
    vtkDebugMacro("vtkSlicerROIGUI: ProcessGUIEvent: Remove ROI Button event: " << event << ".\n");
    // check to see if should confirm
    const char * confirmDelete = ((vtkSlicerApplication *)this->GetApplication())->GetConfirmDelete();
    int confirmDeleteFlag = 0;
    if (confirmDelete != NULL &&
      strncmp(confirmDelete, "1", 1) == 0)
      {
      vtkDebugMacro("vtkSlicerROIGUI: ProcessGUIEvent: confirm delete flag is 1\n");
      confirmDeleteFlag = 1;
      }
    else
      {
      vtkDebugMacro("Not confirming deletes, confirmDelete = '" << confirmDelete << "'\n");
      }
    // save state for undo
    //this->MRMLScene->SaveStateForUndo();

    // get the row that was last selected
    int numRows = this->MultiColumnList->GetWidget()->GetNumberOfSelectedRows();
    if (numRows == 1)
      {
      int row[1];
      this->MultiColumnList->GetWidget()->GetSelectedRows(row);

      if (confirmDeleteFlag)
        {
        // confirm that really want to remove this ROI
        std::cout << "Removing ROI " << row[0] << endl;
        }

      // then remove that ROI by index
      activeROIListNode->RemoveROI(row[0]);
      if (row[0] > 0) 
        {
        vtkMRMLROINode *roi = activeROIListNode->GetNthROINode(row[0]-1);
        this->ROIDisplayWidget->SetROINode(roi);
        }
      }
    else
      {
      vtkErrorMacro ("Selected rows (" << numRows << ") not 1, just pick one to delete for now\n");
      return;
      }
    }

  if (button == this->RemoveROIListButton && event == vtkKWPushButton::InvokedEvent)
    {
    vtkDebugMacro("vtkSlicerROIGUI: ProcessGUIEvent: Remove ROI List Button event: " << event << ".\n");
    // save state for undo
    //this->MRMLScene->SaveStateForUndo();
    activeROIListNode->RemoveAllROIs();
    }

  if (button == this->GetVisibilityToggle()  && event ==  vtkKWPushButton::InvokedEvent)
    {
    vtkDebugMacro("vtkSlicerROIGUI: ProcessGUIEvent: Visibility button event: " << event << ".\n");
    // change the visibility
    activeROIListNode->SetVisibility( ! activeROIListNode->GetVisibility());
    }
  // list colour
  vtkKWChangeColorButton *colorButton = vtkKWChangeColorButton::SafeDownCast(caller);
  if (colorButton == this->ROIColorButton && event == vtkKWChangeColorButton::ColorChangedEvent)
    {
    vtkDebugMacro("ProcessGUIEvents: list colour button change event\n");
    // change the colour
    activeROIListNode->SetColor(this->ROIColorButton->GetColor());
    }
  if (colorButton == this->ROISelectedColorButton && event == vtkKWChangeColorButton::ColorChangedEvent)
    {
    vtkDebugMacro("ProcessGUIEvents: list selected colour button change event\n");
    // change the selected colour
    activeROIListNode->SetSelectedColor(this->ROISelectedColorButton->GetColor());
    }

  // list and text sizes
  vtkKWScaleWithEntry *scale = vtkKWScaleWithEntry::SafeDownCast(caller);
  if (scale == this->ROITextScale && event == vtkKWScale::ScaleValueChangedEvent)
    {
    activeROIListNode->SetTextScale(this->ROITextScale->GetValue());
    }
  else if (scale == this->ROIOpacity && event == vtkKWScale::ScaleValueChangedEvent)
    {
    activeROIListNode->SetOpacity(this->ROIOpacity->GetValue());
    }
  // Position and Size Scale sizes

  //Update GUI according the selected row  
  vtkKWMultiColumnList *Column = vtkKWMultiColumnList::SafeDownCast(caller);
  if (Column == this->MultiColumnList->GetWidget() && event == vtkKWMultiColumnList::SelectionChangedEvent)
    {
    int numRows = this->MultiColumnList->GetWidget()->GetNumberOfSelectedRows();
    if (numRows == 1)
      {
      int row[1];
      this->MultiColumnList->GetWidget()->GetSelectedRows(row);

      vtkMRMLROINode *roi = activeROIListNode->GetNthROINode(row[0]);
      this->ROIDisplayWidget->SetROINode(roi);

      float *xyz;
      float *Radiusxyz;
     
      if(ActiveVolumeNodeID ==NULL)
        {
        xyz = activeROIListNode->GetNthROIXYZ(row[0]);
        Radiusxyz = activeROIListNode->GetNthROIRadiusXYZ(row[0]);
        }
      else
        {
        xyz = activeROIListNode->GetNthROIIJK(row[0]);
        Radiusxyz = activeROIListNode->GetNthROIRadiusIJK(row[0]);
        }

      if(ActiveVolumeNodeID ==NULL)
        {
        activeROIListNode->SetNthROIXYZ(row[0], xyz[0], xyz[1], xyz[2]);
        activeROIListNode->SetNthROIRadiusXYZ(row[0], Radiusxyz[0], Radiusxyz[1], Radiusxyz[2]);
        }
      else
        {
        activeROIListNode->SetNthROIIJK(row[0], xyz[0], xyz[1], xyz[2]);
        activeROIListNode->SetNthROIRadiusIJK(row[0], Radiusxyz[0], Radiusxyz[1], Radiusxyz[2]);
        }
      }
    else if (numRows > 1)
      {
      vtkErrorMacro ("Selected rows (" << numRows << ") not 1, just pick one to delete for now\n");
      return;
      }
    }

  return;
}


//---------------------------------------------------------------------------
void vtkSlicerROIGUI::ProcessLogicEvents ( vtkObject *caller,
                                          unsigned long event, void *callData )
{
  // Fill in
}


//---------------------------------------------------------------------------
void vtkSlicerROIGUI::ProcessMRMLEvents ( vtkObject *caller,
                                         unsigned long event, void *callData )
{
  vtkDebugMacro("vtkSlicerROIsGUI::ProcessMRMLEvents: event = " << event << ".\n");

  if (event == vtkCommand::WidgetValueChangedEvent)
    {
    vtkDebugMacro("got a widget value changed event... the list node was changed.\n");
    }

  // did the selected node get modified?
  if (this->ApplicationLogic)
    {
    vtkMRMLSelectionNode *selnode = this->ApplicationLogic->GetSelectionNode();
    if (selnode != NULL
      && vtkMRMLSelectionNode::SafeDownCast(caller) == selnode
      && event == vtkCommand::ModifiedEvent)
      {
      vtkDebugMacro("The selection node changed\n");
      // is the active roi list id out of synch with our selection?
      if (selnode->GetActiveROIListID() != NULL &&
        this->GetROIListNodeID() != NULL)
        {
        if (strcmp(selnode->GetActiveROIListID(), this->GetROIListNodeID()) != 0)
          {
          vtkDebugMacro("Updating the ROI gui's ROI list node id\n");
          this->SetROIListNodeID(selnode->GetActiveROIListID());
          }
        }
      }
    }

  vtkMRMLROIListNode *node = vtkMRMLROIListNode::SafeDownCast(caller);
  vtkMRMLROIListNode *activeROIListNode = (vtkMRMLROIListNode *)this->MRMLScene->GetNodeByID(this->GetROIListNodeID());

  if (node == activeROIListNode && event == vtkCommand::ModifiedEvent)
    {
    vtkDebugMacro("\tmodified event on the ROI list node.\n");
    if (node == NULL)
      {
      vtkDebugMacro("\tBUT: the node is null\n");
      return;
      }
    vtkDebugMacro("\t\tUpdating the GUI\n");
    // update the table
    SetGUIFromList(activeROIListNode);
    return;
    }

  if (node == activeROIListNode && event == vtkMRMLROIListNode::ROIModifiedEvent)
    {
    vtkDebugMacro("\tROI modified event on the active ROI list.");
    if (node == NULL)
      {
      return;
      }
    SetGUIFromList(activeROIListNode);
    return;
    }

  if (node == activeROIListNode && event == vtkMRMLROIListNode::DisplayModifiedEvent)
    {
    vtkDebugMacro("vtkSlicerROIGUI::ProcessMRMLEvents: DisplayModified event on the ROI list node...\n");
    }

  if (node == vtkMRMLROIListNode::SafeDownCast(this->ROIListSelectorWidget->GetSelected()) && event == vtkCommand::ModifiedEvent)
    {
    vtkDebugMacro("\tmodified event on the ROI list selected node.\n");
    if (activeROIListNode !=  vtkMRMLROIListNode::SafeDownCast(this->ROIListSelectorWidget->GetSelected()))
      {
      // select it first off
      this->SetROIListNode(vtkMRMLROIListNode::SafeDownCast(this->ROIListSelectorWidget->GetSelected()));
      }
    SetGUIFromList(activeROIListNode);
    return;        
  }    

vtkDebugMacro("vtkSlicerROIGUI: Done processing mrml events...");
}

//---------------------------------------------------------------------------
void vtkSlicerROIGUI::UpdateGUI()
{
// Fill in
}

//---------------------------------------------------------------------------
void vtkSlicerROIGUI::SetGUIFromList(vtkMRMLROIListNode * activeROIListNode)
{
  if (activeROIListNode == NULL)
    {
    //clear out the list box
    if (this->MultiColumnList)
      {
      if (this->MultiColumnList->GetWidget()->GetNumberOfRows() != 0)
        {
        this->MultiColumnList->GetWidget()->DeleteAllRows();
        }
      }
    return;
    }

  // Update the volume selector widget 
  vtkMRMLVolumeNode *VolumeNode =
    vtkMRMLVolumeNode::SafeDownCast(this->VolumeNodeSelectorWidget->GetSelected());
  char* ActiveVolumeNodeID = activeROIListNode->GetVolumeNodeID();
  if (VolumeNode)
    {
    if (VolumeNode->GetID() != ActiveVolumeNodeID)
      {
      if (ActiveVolumeNodeID != NULL)
        {
        this->VolumeNodeSelectorWidget->SetSelected((vtkMRMLVolumeNode *)this->MRMLScene->GetNodeByID(ActiveVolumeNodeID)); 
        }
      else 
        {
        this->VolumeNodeSelectorWidget->SetSelected(NULL); 
        this->VolumeNodeSelectorWidget->SetSelected((vtkMRMLVolumeNode *)this->MRMLScene->GetNodeByID(ActiveVolumeNodeID)); 
        }
      this->VolumeNodeSelectorWidget->UpdateMenu();
      }
    }
  else if (ActiveVolumeNodeID != NULL)
    {
    this->VolumeNodeSelectorWidget->SetSelected((vtkMRMLVolumeNode *)this->MRMLScene->GetNodeByID(ActiveVolumeNodeID)); 
    this->VolumeNodeSelectorWidget->UpdateMenu();
    }

  int numPoints = activeROIListNode->GetNumberOfROIs();
  bool deleteFlag = true;
  //Used to set the selected cell and update the xyz position scalewidget
  int CurrentSeletedRow;
  int CurrentSeletedCol;
  int* LastSeletedRow = new int[1];
  int* LastSeletedCol = new int[1];
  if (numPoints == 1)
    {
    CurrentSeletedRow = 0;
    CurrentSeletedCol = 0;
    }
  else if (numPoints > 1)
    {
    this->MultiColumnList->GetWidget()->GetSelectedCells(LastSeletedRow,LastSeletedCol);
    if (LastSeletedRow[0]  > numPoints - 1)
      {
      CurrentSeletedRow = numPoints - 1;
      CurrentSeletedCol = LastSeletedCol[0];
      }
    else
      {
      CurrentSeletedRow  = LastSeletedRow[0];
      CurrentSeletedCol = LastSeletedCol[0];
      }
    }

  delete [] LastSeletedRow;
  delete [] LastSeletedCol;

  if (numPoints != this->MultiColumnList->GetWidget()->GetNumberOfRows())
    {
    vtkDebugMacro("SetGUIFromList: numPoints " << numPoints << " doesn't match number of rows " << this->MultiColumnList->GetWidget()->GetNumberOfRows() << ", so deleting all of them and starting from scratch\n");

    // clear out the multi column list box and fill it in with the
    // new list
    this->MultiColumnList->GetWidget()->DeleteAllRows();
    }
  else
    {
    deleteFlag = false;
    }


  // a row for each point
  float *xyz;
  float *Radiusxyz;

  for (int row = 0; row < numPoints; row++)
    {
    // add a row for this point
    if (deleteFlag)
      {
      vtkDebugMacro("SetGUIFromList: Adding point " << row << " to the table" << endl);
      this->MultiColumnList->GetWidget()->AddRow();
      }
    vtkDebugMacro("SetGUIFromList: getting " << row << "the ROI xyz - total ROIs = " << numPoints);
    // now populate it
    if(ActiveVolumeNodeID == NULL)
      {
      xyz = activeROIListNode->GetNthROIXYZ(row);
      }
    else
      {
      xyz = activeROIListNode->GetNthROIIJK(row);
      }
    if (xyz == NULL)
      {
      vtkErrorMacro ("SetGUIFromList: ERROR: got null xyz for point " << row << endl);
      }

    vtkDebugMacro("Getting nth ROI Radiusxyz"); 
    if(ActiveVolumeNodeID == NULL)
      {
      Radiusxyz = activeROIListNode->GetNthROIRadiusXYZ(row);
      }
    else
      {
      Radiusxyz = activeROIListNode->GetNthROIRadiusIJK(row);
      }

    if (activeROIListNode->GetNthROILabelText(row) != NULL)
      {
      if (strcmp(this->MultiColumnList->GetWidget()->GetCellText(row,this->NameColumn), activeROIListNode->GetNthROILabelText(row)) != 0)
        {
        this->MultiColumnList->GetWidget()->SetCellText(row,this->NameColumn,activeROIListNode->GetNthROILabelText(row));
        }               
      }
    else
      {
      if (strcmp(this->MultiColumnList->GetWidget()->GetCellText(row,this->NameColumn), "(none)") != 0)
        {
        this->MultiColumnList->GetWidget()->SetCellText(row,this->NameColumn,"(none)");
        }
      }

    // selected
    if (deleteFlag || this->MultiColumnList->GetWidget()->GetCellTextAsInt(row,this->SelectedColumn) != (activeROIListNode->GetNthROISelected(row) ? 1 : 0))
      {
      this->MultiColumnList->GetWidget()->SetCellTextAsInt(row,this->SelectedColumn,(activeROIListNode->GetNthROISelected(row) ? 1 : 0));
      this->MultiColumnList->GetWidget()->SetCellWindowCommandToCheckButton(row,this->SelectedColumn);
      }

    if (xyz != NULL)
      {
      // always set it if it's a new row added because all were
      // deleted, because the numerical default is 0
      if (deleteFlag || this->MultiColumnList->GetWidget()->GetCellTextAsDouble(row,this->XColumn) != xyz[0])
        {
        this->MultiColumnList->GetWidget()->SetCellTextAsDouble(row,this->XColumn,xyz[0]);
        } 
      if (deleteFlag || this->MultiColumnList->GetWidget()->GetCellTextAsDouble(row,this->YColumn) != xyz[1])
        {
        this->MultiColumnList->GetWidget()->SetCellTextAsDouble(row,this->YColumn,xyz[1]);
        }
      if (deleteFlag || this->MultiColumnList->GetWidget()->GetCellTextAsDouble(row,this->ZColumn) != xyz[2])
        {
        this->MultiColumnList->GetWidget()->SetCellTextAsDouble(row,this->ZColumn,xyz[2]);
        }
      }

    if (Radiusxyz != NULL)
      {
      if (deleteFlag || this->MultiColumnList->GetWidget()->GetCellTextAsDouble(row,this->RadiusXColumn) != Radiusxyz[0])
        {
        this->MultiColumnList->GetWidget()->SetCellTextAsDouble(row,this->RadiusXColumn,Radiusxyz[0]);
        } 
      if (deleteFlag || this->MultiColumnList->GetWidget()->GetCellTextAsDouble(row,this->RadiusYColumn) != Radiusxyz[1])
        {
        this->MultiColumnList->GetWidget()->SetCellTextAsDouble(row,this->RadiusYColumn,Radiusxyz[1]);
        }
      if (deleteFlag || this->MultiColumnList->GetWidget()->GetCellTextAsDouble(row,this->RadiusZColumn) != Radiusxyz[2])
        {
        this->MultiColumnList->GetWidget()->SetCellTextAsDouble(row,this->RadiusZColumn,Radiusxyz[2]);
        } 
      }
    }

  if (numPoints >=1)
    {
    this->MultiColumnList->GetWidget()->SelectCell(CurrentSeletedRow,CurrentSeletedCol);
    }

  vtkDebugMacro("Now going to update GUI from the logic's active list");

  // update the visibility, color, scale buttons to match the displayed list's
  if (activeROIListNode == NULL)
    {
    vtkErrorMacro ("vtkSlicerROIGUI::SetGUIFromList: ERROR: no active ROI list node in the gui class!\n");                
    return;
    }

  // Update the label text: 
  // No volume selected IJK coordinates, 
  // Volume selected RAS coordinate
  if (ActiveVolumeNodeID == NULL)
    {
    this->MultiColumnList->GetWidget()->SetColumnTitle (2, "X");
    this->MultiColumnList->GetWidget()->SetColumnTitle (3, "Y");
    this->MultiColumnList->GetWidget()->SetColumnTitle (4, "Z");
    this->MultiColumnList->GetWidget()->SetColumnTitle (5,"X Radius");
    this->MultiColumnList->GetWidget()->SetColumnTitle (6,"Y Radius");
    this->MultiColumnList->GetWidget()->SetColumnTitle (7,"Z Radius");

    }
  else 
    {
    this->MultiColumnList->GetWidget()->SetColumnTitle(2, "I");
    this->MultiColumnList->GetWidget()->SetColumnTitle(3, "J");
    this->MultiColumnList->GetWidget()->SetColumnTitle(4, "K");
    this->MultiColumnList->GetWidget()->SetColumnTitle(5, "I Radius");
    this->MultiColumnList->GetWidget()->SetColumnTitle(6, "J Radius");
    this->MultiColumnList->GetWidget()->SetColumnTitle(7, "K Radius");

    //Update the range according the volume size
    VolumeNode = vtkMRMLVolumeNode::SafeDownCast(this->VolumeNodeSelectorWidget->GetSelected());
    int* dims = new int[3];
    VolumeNode->GetImageData()->GetDimensions(dims);
    delete [] dims;
    }

  vtkDebugMacro(<< "\tupdating the x, y, z location and deltea x, y, z \n");
  //update the xyz position and xyz Radius according the selected row
  int numRows = this->MultiColumnList->GetWidget()->GetNumberOfSelectedRows();
  if (numRows == 1)
    {
    int row[1];
    this->MultiColumnList->GetWidget()->GetSelectedRows(row);
    //float* xyz;
    //float* Radiusxyz;
    if (ActiveVolumeNodeID == NULL)
      {
      xyz = activeROIListNode->GetNthROIXYZ(row[0]);
      Radiusxyz = activeROIListNode->GetNthROIRadiusXYZ(row[0]);
      }
    else
      {
      xyz = activeROIListNode->GetNthROIIJK(row[0]);
      Radiusxyz = activeROIListNode->GetNthROIRadiusIJK(row[0]);
      }
    }

  if (this->GetVisibilityToggle() != NULL &&
    this->GetVisibilityIcons() != NULL)
    {
    if (activeROIListNode->GetVisibility() > 0)
      {
      this->GetVisibilityToggle()->SetImageToIcon(
        this->GetVisibilityIcons()->GetVisibleIcon());
      }
    else
      {
      this->GetVisibilityToggle()->SetImageToIcon(
        this->GetVisibilityIcons()->GetInvisibleIcon());
      }
    }
  else
    {
    vtkErrorMacro ("ERROR; trying up update null visibility toggle!\n");
    }

  // color
  vtkDebugMacro(<< "\tupdating the colour\n");
  double *nodeColor = activeROIListNode->GetColor();
  if (this->ROIColorButton != NULL)
    {
    double *buttonColor = this->ROIColorButton->GetColor();
    if (nodeColor != NULL && buttonColor != NULL && 
      (nodeColor[0] != buttonColor[0] ||
      nodeColor[1] != buttonColor[1] ||
      nodeColor[2] != buttonColor[2]))
      {
      vtkDebugMacro("Updating list color button\n");
      this->ROIColorButton->SetColor(nodeColor);
      }
    }
  else
    {
    vtkErrorMacro("No colour button!\n");
    }

  // selected color
  vtkDebugMacro(<< "\tupdating the selected colour\n");
  double *nodeSelectedColor = activeROIListNode->GetSelectedColor();
  if (this->ROISelectedColorButton != NULL)
    {
    double *buttonSelectedColor = this->ROISelectedColorButton->GetColor();
    if (nodeSelectedColor != NULL && buttonSelectedColor != NULL && 
      (nodeSelectedColor[0] != buttonSelectedColor[0] ||
      nodeSelectedColor[1] != buttonSelectedColor[1] ||
      nodeSelectedColor[2] != buttonSelectedColor[2]))
      {
      vtkDebugMacro("Updating list selected color button\n");
      this->ROISelectedColorButton->SetColor(nodeSelectedColor);
      }
    }
  else
    {
    vtkErrorMacro("No selected colour button!\n");
    }

  // text scale
  vtkDebugMacro(<< "\tupdating the text scale.");
  double scale  = activeROIListNode->GetTextScale();
  if (this->ROITextScale != NULL &&
    scale != this->ROITextScale->GetValue())
    {
    this->ROITextScale->SetValue(scale);
    }

  // opacity
  vtkDebugMacro(<< "\tupdating the opacity");
  scale = activeROIListNode->GetOpacity();
  if (this->ROIOpacity != NULL &&
    scale != this->ROIOpacity->GetValue())
    {
    this->ROIOpacity->SetValue(scale);
    }

  
  return;
}

//---------------------------------------------------------------------------
void vtkSlicerROIGUI::CreateModuleEventBindings ( )
{
// Fill in
}

//---------------------------------------------------------------------------
void vtkSlicerROIGUI::ReleaseModuleEventBindings ( )
{
// Fill in
}


//---------------------------------------------------------------------------
void vtkSlicerROIGUI::Enter ( )
{
  if ( this->Built == false )
    {
    this->BuildGUI();
    this->Built = true;
    this->AddGUIObservers();
    }
  this->CreateModuleEventBindings();
  this->UpdateGUI();
}

//---------------------------------------------------------------------------
void vtkSlicerROIGUI::Exit ( )
{
  this->ReleaseModuleEventBindings();
}


//---------------------------------------------------------------------------
void vtkSlicerROIGUI::TearDownGUI ( )
{
  this->Exit();
  if ( this->Built )
    {
    this->RemoveGUIObservers();
    }
}


//---------------------------------------------------------------------------
void vtkSlicerROIGUI::BuildGUI ( )
{
  vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();

  // create a page
  this->UIPanel->AddPage ( "ROI", "ROI", NULL );

  // Define your help text and build the help frame here.
  const char *help = "The ROI Module creates and manages ROI. ";
  const char *about = "This work was supported by NA-MIC, NAC, BIRN, NCIGT, and the Slicer Community. See <a>http://www.slicer.org</a> for details. ";
  vtkKWWidget *page = this->UIPanel->GetPageWidget ( "ROI" );
  this->BuildHelpAndAboutFrame ( page, help, about );

  // ---
  // ROI FRAME
  vtkSlicerModuleCollapsibleFrame *ROIFrame = vtkSlicerModuleCollapsibleFrame::New();
  ROIFrame->SetParent( page );
  ROIFrame->Create();
  ROIFrame->SetLabelText("ROI Display");
  ROIFrame->ExpandFrame();
  app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
    ROIFrame->GetWidgetName(), this->UIPanel->GetPageWidget("ROI")->GetWidgetName());

  // node selector
  this->ROISelectorWidget = vtkSlicerNodeSelectorWidget::New();
  this->ROISelectorWidget->SetParent(ROIFrame->GetFrame());
  this->ROISelectorWidget->Create();
  this->ROISelectorWidget->SetNodeClass("vtkMRMLROINode", NULL, NULL, NULL);
  this->ROISelectorWidget->NewNodeEnabledOn();
  this->ROISelectorWidget->SetMRMLScene(this->GetMRMLScene());
  this->ROISelectorWidget->SetBorderWidth(2);
  this->ROISelectorWidget->SetPadX(2);
  this->ROISelectorWidget->SetPadY(2);
  this->ROISelectorWidget->SetShowHidden(1);
  //this->ROISelectorWidget->GetWidget()->IndicatorVisibilityOff();
  this->ROISelectorWidget->GetWidget()->SetWidth(24);
  this->ROISelectorWidget->SetLabelText( "ROI Select: ");
  this->ROISelectorWidget->SetBalloonHelpString("Select a ROI from the current mrml scene.");
  this->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                this->ROISelectorWidget->GetWidgetName(),
                ROIFrame->GetFrame()->GetWidgetName() );

  // display widgwt
  this->ROIDisplayWidget = vtkSlicerROIDisplayWidget::New();
  this->ROIDisplayWidget->SetParent ( ROIFrame->GetFrame() );
  this->ROIDisplayWidget->Create();
  this->Script("pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s", 
               this->ROIDisplayWidget->GetWidgetName(),
               ROIFrame->GetFrame()->GetWidgetName());



  // ---
  // LIST FRAME
  vtkSlicerModuleCollapsibleFrame *ROIListFrame = vtkSlicerModuleCollapsibleFrame::New();
  ROIListFrame->SetParent( page );
  ROIListFrame->Create();
  ROIListFrame->SetLabelText("ROI List (DEPRICATED)");
  //ROIListFrame->ExpandFrame();
  ROIListFrame->CollapseFrame();
  app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
    ROIListFrame->GetWidgetName(), this->UIPanel->GetPageWidget("ROI")->GetWidgetName());

  // node selector
  this->ROIListSelectorWidget = vtkSlicerNodeSelectorWidget::New();
  this->ROIListSelectorWidget->SetParent(ROIListFrame->GetFrame());
  this->ROIListSelectorWidget->Create();
  this->ROIListSelectorWidget->SetNodeClass("vtkMRMLROIListNode", NULL, NULL, NULL);
  this->ROIListSelectorWidget->NewNodeEnabledOn();
  this->ROIListSelectorWidget->SetMRMLScene(this->GetMRMLScene());
  this->ROIListSelectorWidget->SetBorderWidth(2);
  this->ROIListSelectorWidget->SetPadX(2);
  this->ROIListSelectorWidget->SetPadY(2);
  //this->ROIListSelectorWidget->GetWidget()->IndicatorVisibilityOff();
  this->ROIListSelectorWidget->GetWidget()->SetWidth(24);
  this->ROIListSelectorWidget->SetLabelText( "ROI List Select: ");
  this->ROIListSelectorWidget->SetBalloonHelpString("Select a ROI list from the current mrml scene.");
  this->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
    this->ROIListSelectorWidget->GetWidgetName());

  // Vlolume node selector
  this->VolumeNodeSelectorWidget = vtkSlicerNodeSelectorWidget::New();
  this->VolumeNodeSelectorWidget->SetParent(ROIListFrame->GetFrame());
  this->VolumeNodeSelectorWidget->Create();
  this->VolumeNodeSelectorWidget->SetNodeClass("vtkMRMLVolumeNode", NULL, NULL, NULL);
  this->VolumeNodeSelectorWidget->SetMRMLScene(this->GetMRMLScene());
  this->VolumeNodeSelectorWidget->SetSelected(NULL); // force empty select
  this->VolumeNodeSelectorWidget->SetNoneEnabled(1);
  this->VolumeNodeSelectorWidget->UpdateMenu();
  this->VolumeNodeSelectorWidget->SetBorderWidth(2);
  this->VolumeNodeSelectorWidget->SetPadX(2);
  this->VolumeNodeSelectorWidget->SetPadY(2);
  this->VolumeNodeSelectorWidget->GetWidget()->SetWidth(24);
  this->VolumeNodeSelectorWidget->SetLabelText( "Volume Node Select: ");
  this->VolumeNodeSelectorWidget->SetBalloonHelpString("Select a volume node associated with the ROI.");
  this->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
    this->VolumeNodeSelectorWidget->GetWidgetName());

  // add the multicolumn list to show the points
  this->MultiColumnList = vtkKWMultiColumnListWithScrollbars::New ( );
  this->MultiColumnList->SetParent ( ROIListFrame->GetFrame() );
  this->MultiColumnList->Create ( );
  this->MultiColumnList->SetHeight(4);
  this->MultiColumnList->GetWidget()->SetSelectionTypeToCell();
  this->MultiColumnList->GetWidget()->MovableRowsOff();
  this->MultiColumnList->GetWidget()->MovableColumnsOff();

  // set up the columns of data for each point
  // refer to the header file for order
  this->MultiColumnList->GetWidget()->AddColumn("Name");
  this->MultiColumnList->GetWidget()->AddColumn("Selected");
  this->MultiColumnList->GetWidget()->AddColumn("X");
  this->MultiColumnList->GetWidget()->AddColumn("Y");
  this->MultiColumnList->GetWidget()->AddColumn("Z");
  this->MultiColumnList->GetWidget()->AddColumn("X Radius");
  this->MultiColumnList->GetWidget()->AddColumn("Y Radius");
  this->MultiColumnList->GetWidget()->AddColumn("Z Radius");

  // make the selected column editable by checkbox
  this->MultiColumnList->GetWidget()->SetColumnEditWindowToCheckButton(this->SelectedColumn);

  // now set the attributes that are equal across the columns
  int col;
  for (col = 0; col < this->NumberOfColumns; col++)
    {        
    this->MultiColumnList->GetWidget()->SetColumnWidth(col, 6);
    this->MultiColumnList->GetWidget()->SetColumnAlignmentToLeft(col);
    this->MultiColumnList->GetWidget()->ColumnEditableOn(col);
    if (col >= this->XColumn && col <= this->RadiusZColumn)
      {
      this->MultiColumnList->GetWidget()->SetColumnEditWindowToSpinBox(col);
      }
    }

  // set the name column width to be higher
  this->MultiColumnList->GetWidget()->SetColumnWidth(this->NameColumn, 15);
  // set the selected column width a bit higher
  this->MultiColumnList->GetWidget()->SetColumnWidth(this->SelectedColumn, 9);
  app->Script ( "pack %s -fill both -expand true",
    this->MultiColumnList->GetWidgetName());
  this->MultiColumnList->GetWidget()->SetCellUpdatedCommand(this, "UpdateElement");

  // button frame
  vtkKWFrame *buttonFrame = vtkKWFrame::New();
  buttonFrame->SetParent ( ROIListFrame->GetFrame() );
  buttonFrame->Create ( );
  app->Script ("pack %s -side top -anchor nw -fill x -pady 0 -in %s",
    buttonFrame->GetWidgetName(),
    ROIListFrame->GetFrame()->GetWidgetName());

  // add an add ROI button
  this->AddROIButton = vtkKWPushButton::New ( );
  this->AddROIButton->SetParent ( buttonFrame );
  this->AddROIButton->Create ( );
  this->AddROIButton->SetText ("Add ROI");
  this->AddROIButton->SetBalloonHelpString("Add a ROI to the current list");

  // add a remove ROI button
  this->RemoveROIButton = vtkKWPushButton::New ( );
  this->RemoveROIButton->SetParent ( buttonFrame );
  this->RemoveROIButton->Create ( );
  this->RemoveROIButton->SetText ("Remove ROI");
  this->RemoveROIButton->SetBalloonHelpString("Remove the last ROI that was clicked on in the table from the list.");


  // add a remove all ROI from this list button
  this->RemoveROIListButton = vtkKWPushButton::New ( );
  this->RemoveROIListButton->SetParent ( buttonFrame );
  this->RemoveROIListButton->Create ( );
  this->RemoveROIListButton->SetText ("Remove All ROIs");
  this->RemoveROIListButton->SetBalloonHelpString("Remove all ROIs from the list.");

  app->Script("pack %s %s %s -side left -anchor w -padx 4 -pady 2", 
    this->AddROIButton->GetWidgetName(),
    this->RemoveROIButton->GetWidgetName(),
    this->RemoveROIListButton->GetWidgetName());

  // ---
  // DISPLAY FRAME            
  vtkSlicerModuleCollapsibleFrame *displayFrame = vtkSlicerModuleCollapsibleFrame::New ( );
  displayFrame->SetParent ( page );
  displayFrame->Create ( );
  displayFrame->SetLabelText ("ROI List Display (DEPRICATED)");
  //displayFrame->ExpandFrame ( );
  displayFrame->CollapseFrame();
  app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
    displayFrame->GetWidgetName(),
    this->UIPanel->GetPageWidget("ROI")->GetWidgetName());

  // XPosition frame
  vtkKWFrame *XPositionFrame = vtkKWFrame::New();
  XPositionFrame->SetParent ( displayFrame->GetFrame() );
  XPositionFrame->Create ( );
  app->Script ("pack %s -side top -anchor nw -fill x -pady 0 -in %s",
    XPositionFrame->GetWidgetName(),
    displayFrame->GetFrame()->GetWidgetName());

  // scale frame
  vtkKWFrame *scaleFrame = vtkKWFrame::New();
  scaleFrame->SetParent ( displayFrame->GetFrame() );
  scaleFrame->Create ( );
  app->Script ("pack %s -side top -anchor nw -fill x -pady 0 -in %s",
    scaleFrame->GetWidgetName(),
    displayFrame->GetFrame()->GetWidgetName());

  // visibility
  this->VisibilityIcons = vtkSlicerVisibilityIcons::New ( );
  this->VisibilityToggle = vtkKWPushButton::New();
  this->VisibilityToggle->SetParent ( scaleFrame );
  this->VisibilityToggle->Create ( );
  this->VisibilityToggle->SetReliefToFlat ( );
  this->VisibilityToggle->SetOverReliefToNone ( );
  this->VisibilityToggle->SetBorderWidth ( 0 );
  this->VisibilityToggle->SetImageToIcon ( this->VisibilityIcons->GetInvisibleIcon ( ) );        
  this->VisibilityToggle->SetBalloonHelpString ( "Toggles ROI list visibility in the MainViewer." );
  this->VisibilityToggle->SetText ("Visibility");

  // opacity
  this->ROIOpacity = vtkKWScaleWithEntry::New();
  this->ROIOpacity->SetParent( scaleFrame );
  this->ROIOpacity->Create();
  this->ROIOpacity->SetLabelText("Opacity:");
  this->ROIOpacity->SetBalloonHelpString ( "Set the opacity of the ROI list symbols.");
  this->ROIOpacity->GetWidget()->SetRange(0.0, 1.0);
  this->ROIOpacity->GetWidget()->SetOrientationToHorizontal();
  this->ROIOpacity->GetWidget()->SetResolution(0.1);
  this->ROIOpacity->SetEntryWidth(4);

  // text scale
  this->ROITextScale = vtkKWScaleWithEntry::New();
  this->ROITextScale->SetParent( scaleFrame );
  this->ROITextScale->Create();
  this->ROITextScale->SetLabelText("Text Scale:");
  this->ROITextScale->SetBalloonHelpString ( "Set the scale of the ROI text.");
  this->ROITextScale->GetWidget()->SetRange(0.0, 20.0);
  this->ROITextScale->GetWidget()->SetOrientationToHorizontal ();
  this->ROITextScale->GetWidget()->SetResolution(0.5);
  this->ROITextScale->SetEntryWidth(4);

  app->Script("pack %s %s %s -side left -anchor w -padx 2 -pady 2 -in %s", 
    this->VisibilityToggle->GetWidgetName(), this->ROIOpacity->GetWidgetName(),this->ROITextScale->GetWidgetName(),scaleFrame->GetWidgetName() );

  // colour frame
  vtkKWFrame *colourFrame = vtkKWFrame::New();
  colourFrame->SetParent ( displayFrame->GetFrame() );
  colourFrame->Create ( );
  app->Script ("pack %s -side top -anchor nw -fill x -pady 0 -in %s",
    colourFrame->GetWidgetName(),
    displayFrame->GetFrame()->GetWidgetName());

  // color
  this->ROIColorButton = vtkKWChangeColorButton::New();
  this->ROIColorButton->SetParent( colourFrame );
  this->ROIColorButton->Create();
  this->ROIColorButton->SetBorderWidth(0);
  this->ROIColorButton->SetBalloonHelpString("Change the colour of the ROI and text in the MainViewer");
  this->ROIColorButton->SetDialogTitle("List symbol and text color");
  this->ROIColorButton->SetLabelText("Set Color");

  // selected colour
  this->ROISelectedColorButton = vtkKWChangeColorButton::New();
  this->ROISelectedColorButton->SetParent( colourFrame );
  this->ROISelectedColorButton->Create();
  this->ROISelectedColorButton->SetBorderWidth(0);
  this->ROISelectedColorButton->SetBalloonHelpString("Change the color of the selected ROI symbols and text in the MainViewer");

  this->ROISelectedColorButton->SetDialogTitle("List selected symbol and text color");
  this->ROISelectedColorButton->SetLabelText("Set Selected Color");

  // pack the colours 
  app->Script("pack %s %s -side left -anchor w -padx 4 -pady 2 -in %s",
    this->ROIColorButton->GetWidgetName(), this->ROISelectedColorButton->GetWidgetName(),
    colourFrame->GetWidgetName());

  buttonFrame->Delete ();
  ROIFrame->Delete ();
  ROIListFrame->Delete ();
  displayFrame->Delete ();
  scaleFrame->Delete ();
  colourFrame->Delete ();
  XPositionFrame->Delete ();
   return;
}

void vtkSlicerROIGUI::UpdateElement(int row, int col, char * str)
 {
  vtkDebugMacro("UpdateElement: row = " << row << ", col = " << col << ", str = " << str << "\n");

  // make sure that the row and column exists in the table
  if ((row >= 0) && (row < this->MultiColumnList->GetWidget()->GetNumberOfRows()) &&
    (col >= 0) && (col < this->MultiColumnList->GetWidget()->GetNumberOfColumns()))
    {
    vtkMRMLROIListNode *activeROIListNode = (vtkMRMLROIListNode *)this->MRMLScene->GetNodeByID(this->GetROIListNodeID());
    // is there an active list?
    if (activeROIListNode == NULL)
      {
      // 
      vtkErrorMacro ("UpdateElement: ERROR: No ROI List, add one first!\n");
      return;
      }

    char *ActiveVolumeNodeID = activeROIListNode->GetVolumeNodeID();
    // now update the requested value
    if (col == this->NameColumn)
      {
      activeROIListNode->SetNthROILabelText(row, str);
      }
    else if (col == this->SelectedColumn)
      {
      // selected
      vtkDebugMacro("UpdateElement: setting node " <<  activeROIListNode->GetNthROILabelText(row) << "'s selected flag to " << str << endl);
      activeROIListNode->SetNthROISelected(row, (atoi(str) == 1));
      }
    else if (col >= this->XColumn && col <= this->RadiusZColumn)
      {
      // get the current xyz
      float * xyz;
      float * Radiusxyz;
      if (ActiveVolumeNodeID == NULL)
        {
        xyz = activeROIListNode->GetNthROIXYZ(row);
        Radiusxyz = activeROIListNode->GetNthROIRadiusXYZ(row);
        }
      else
        {
        xyz = activeROIListNode->GetNthROIIJK(row);
        Radiusxyz = activeROIListNode->GetNthROIRadiusIJK(row);
        }
     
      // now set the new one
      float newCoordinate = atof(str);
      if (col == this->XColumn) 
        { 
        if (ActiveVolumeNodeID == NULL)
          {
          activeROIListNode->SetNthROIXYZ(row, newCoordinate, xyz[1], xyz[2]); 
          }
        else
          {
          activeROIListNode->SetNthROIIJK(row, newCoordinate, xyz[1], xyz[2]); 
          }
        }
      if (col == this->YColumn) 
        { 
        if (ActiveVolumeNodeID == NULL)
          {
          activeROIListNode->SetNthROIXYZ(row, xyz[0], newCoordinate, xyz[2]);
          }
        else
          {
          activeROIListNode->SetNthROIIJK(row, xyz[0], newCoordinate, xyz[2]);
          }
        }
      if (col == this->ZColumn) 
        { 
        if (ActiveVolumeNodeID == NULL)
          {
          activeROIListNode->SetNthROIXYZ(row, xyz[0], xyz[1], newCoordinate);
          }
        else
          {
          activeROIListNode->SetNthROIIJK(row, xyz[0], xyz[1], newCoordinate);
          }
        }
      if (col == this->RadiusXColumn) 
        { 
        if (ActiveVolumeNodeID == NULL)
          {
          activeROIListNode->SetNthROIRadiusXYZ(row, newCoordinate, Radiusxyz[1], Radiusxyz[2]); 
          }
        else
          {
          activeROIListNode->SetNthROIRadiusIJK(row, newCoordinate, Radiusxyz[1], Radiusxyz[2]); 
          }
        }
      if (col == this->RadiusYColumn) 
        { 
        if (ActiveVolumeNodeID == NULL)
          {
          activeROIListNode->SetNthROIRadiusXYZ(row, Radiusxyz[0], newCoordinate, Radiusxyz[2]);  
          }
        else
          {
          activeROIListNode->SetNthROIRadiusIJK(row, Radiusxyz[0], newCoordinate, Radiusxyz[2]); 
          }
        }
      if (col == this->RadiusZColumn) 
        {
        if (ActiveVolumeNodeID == NULL)
          {
          activeROIListNode->SetNthROIRadiusXYZ(row, Radiusxyz[0], Radiusxyz[1], newCoordinate);
          }
        else
          {
          activeROIListNode->SetNthROIRadiusIJK(row, Radiusxyz[0], Radiusxyz[1], newCoordinate);
          }
        }
      }
    else
      {
      vtkErrorMacro ("UpdateElement: ERROR: invalid column number " << col << ", valid values are 0-" << this->NumberOfColumns << endl);
      return;
      }
    }
}

//---------------------------------------------------------------------------
void vtkSlicerROIGUI::SetROIListNode (vtkMRMLROIListNode *ROIListNode)
{
  if (ROIListNode == NULL)
    {
    vtkErrorMacro ("ERROR: SetROIListNode - list node is null.\n");
    return;
    }
  // save the ID
  vtkDebugMacro("setting the ROI list node id to " << ROIListNode->GetID());
  this->SetROIListNodeID(ROIListNode->GetID());
  if (this->ROIListNode && this->ROIListNode->GetNthROINode(0))
    {
    this->ROIDisplayWidget->SetROINode(ROIListNode->GetNthROINode(0));
    }
  return;
}

//---------------------------------------------------------------------------
void vtkSlicerROIGUI::SetROIListNodeID (char * id)
{
  if (this->GetROIListNodeID() != NULL &&
    id != NULL &&
    strcmp(id,this->GetROIListNodeID()) == 0)
    {
    vtkDebugMacro("no change in id, not doing anything for now: " << id << endl);
    return;
    }

  // set the id properly - see the vtkSetStringMacro
  this->ROIListNodeID = id;

  if (id == NULL)
    {
    vtkDebugMacro("SetROIListNodeID: NULL input id, removed observers and returning.\n");
    return;
    }

  // get the new node
  vtkMRMLROIListNode *ROIlist = vtkMRMLROIListNode::SafeDownCast(this->MRMLScene->GetNodeByID(this->GetROIListNodeID()));

  // set up observers on the new node
  if (ROIlist != NULL)
    {
    vtkSetAndObserveMRMLObjectMacro(this->ROIListNode, NULL);
    vtkIntArray *events = vtkIntArray::New();
    events->InsertNextValue(vtkCommand::ModifiedEvent);
    events->InsertNextValue(vtkMRMLROIListNode::DisplayModifiedEvent);
    events->InsertNextValue(vtkMRMLROIListNode::ROIModifiedEvent);
    vtkSetAndObserveMRMLNodeEventsMacro(this->ROIListNode, ROIlist, events);
    events->Delete();
    // set up the GUI
    this->SetGUIFromList(this->ROIListNode); 
    }
  else
    {
    vtkErrorMacro ("ERROR: unable to get the mrml ROI node to observe!\n");
    }

  // update the selected ROI list id
  if (this->ApplicationLogic != NULL &&
    this->ApplicationLogic->GetSelectionNode() != NULL &&
    this->ROIListNodeID != NULL)
    {
    vtkDebugMacro("ROI GUI: setting the active ROI list id to " << this->ROIListNodeID);
    this->ApplicationLogic->GetSelectionNode()->SetActiveROIListID( this->ROIListNodeID );
    }
  return;
}
