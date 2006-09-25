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

#include "vtkSlicerFiducialsGUI.h"

//---------------------------------------------------------------------------
vtkStandardNewMacro (vtkSlicerFiducialsGUI );
vtkCxxRevisionMacro ( vtkSlicerFiducialsGUI, "$Revision: 1.0 $");


//---------------------------------------------------------------------------
vtkSlicerFiducialsGUI::vtkSlicerFiducialsGUI ( )
{
    this->Logic = NULL;
    this->FiducialListSelectorWidget = NULL;
    this->FiducialListNodeID = NULL; // "(none)";

    this->AddFiducialButton = NULL;
    this->RemoveFiducialButton = NULL;
    this->RemoveFiducialsButton = NULL;
    
    this->VisibilityToggle = NULL;
    this->VisibilityIcons = NULL;

    this->ListColorButton = NULL;
    this->ListSelectedColorButton = NULL;
    this->ListSymbolScale = NULL;
    this->ListTextScale = NULL;
    this->ListOpacity = NULL;
    
    this->MultiColumnList = NULL;
    this->NumberOfColumns = 9;
}


//---------------------------------------------------------------------------
vtkSlicerFiducialsGUI::~vtkSlicerFiducialsGUI ( )
{

    this->SetModuleLogic ( NULL );

    if (this->FiducialListSelectorWidget)
    {
        this->FiducialListSelectorWidget->SetParent(NULL);
        this->FiducialListSelectorWidget->Delete();
        this->FiducialListSelectorWidget = NULL;
    }
     
    if (this->AddFiducialButton ) {
        this->AddFiducialButton->SetParent (NULL );
        this->AddFiducialButton->Delete ( );
        this->AddFiducialButton = NULL;
    }

    if (this->RemoveFiducialButton ) {
        this->RemoveFiducialButton->SetParent (NULL );
        this->RemoveFiducialButton->Delete ( );
        this->RemoveFiducialButton = NULL;
    }

    if (this->RemoveFiducialsButton ) {
        this->RemoveFiducialsButton->SetParent (NULL );
        this->RemoveFiducialsButton->Delete ( );
        this->RemoveFiducialsButton = NULL;
    }

    if (this->VisibilityToggle) {
        this->VisibilityToggle->SetParent(NULL);
        this->VisibilityToggle->Delete();
        this->VisibilityToggle = NULL;
    }

    if ( this->VisibilityIcons ) {
        this->VisibilityIcons->Delete  ( );
        this->VisibilityIcons = NULL;
    }

    if (this->ListColorButton) {
        this->ListColorButton->SetParent(NULL);
        this->ListColorButton->Delete();
        this->ListColorButton = NULL;
    }

    if (this->ListSelectedColorButton) {
        this->ListSelectedColorButton->SetParent(NULL);
        this->ListSelectedColorButton->Delete();
        this->ListSelectedColorButton = NULL;
    }

    if (this->ListSymbolScale) {
        this->ListSymbolScale->SetParent(NULL);
        this->ListSymbolScale->Delete();
        this->ListSymbolScale = NULL;
    }

    if (this->ListTextScale) {
        this->ListTextScale->SetParent(NULL);
        this->ListTextScale->Delete();
        this->ListTextScale = NULL;
    }

    if (this->ListOpacity) {
        this->ListOpacity->SetParent(NULL);
        this->ListOpacity->Delete();
        this->ListOpacity = NULL;
    }

    if (this->MultiColumnList) {
      this->MultiColumnList->SetParent(NULL);
      this->MultiColumnList->Delete();
      this->MultiColumnList = NULL;
    }

    //this->SetFiducialListNodeID("(none)");
    this->SetFiducialListNodeID(NULL);
}


//---------------------------------------------------------------------------
void vtkSlicerFiducialsGUI::PrintSelf ( ostream& os, vtkIndent indent )
{
    this->vtkObject::PrintSelf ( os, indent );

    os << indent << "SlicerFiducialsGUI: " << this->GetClassName ( ) << "\n";
    os << indent << "Logic: " << this->GetLogic ( ) << "\n";
    // print widgets?
}



//---------------------------------------------------------------------------
void vtkSlicerFiducialsGUI::RemoveGUIObservers ( )
{
    vtkDebugMacro("vtkSlicerFiducialsGUI: RemoveGUIObservers\n");
    this->FiducialListSelectorWidget->RemoveObservers(vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand );
    this->AddFiducialButton->RemoveObservers ( vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
    this->RemoveFiducialButton->RemoveObservers ( vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
    this->RemoveFiducialsButton->RemoveObservers ( vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );

    this->VisibilityToggle->RemoveObservers (vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
    this->ListColorButton->RemoveObservers (vtkKWChangeColorButton::ColorChangedEvent, (vtkCommand *)this->GUICallbackCommand);
    this->ListSelectedColorButton->RemoveObservers (vtkKWChangeColorButton::ColorChangedEvent, (vtkCommand *)this->GUICallbackCommand);
    this->ListSymbolScale->RemoveObservers (vtkKWScale::ScaleValueChangedEvent, (vtkCommand *)this->GUICallbackCommand);
    this->ListTextScale->RemoveObservers (vtkKWScale::ScaleValueChangedEvent, (vtkCommand *)this->GUICallbackCommand);
    this->ListOpacity->RemoveObservers (vtkKWScale::ScaleValueChangedEvent, (vtkCommand *)this->GUICallbackCommand);
    
    this->RemoveObservers (vtkSlicerFiducialsGUI::FiducialListIDModifiedEvent, (vtkCommand *)this->GUICallbackCommand);
}


//---------------------------------------------------------------------------
void vtkSlicerFiducialsGUI::AddGUIObservers ( )
{
    vtkDebugMacro("vtkSlicerFiducialsGUI: AddGUIObservers\n");
    this->FiducialListSelectorWidget->AddObserver(vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand );  
    this->AddFiducialButton->AddObserver ( vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );
    this->RemoveFiducialButton->AddObserver ( vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );
    this->RemoveFiducialsButton->AddObserver ( vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );
    
    this->VisibilityToggle->AddObserver (vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );
    this->ListColorButton->AddObserver (vtkKWChangeColorButton::ColorChangedEvent, (vtkCommand *)this->GUICallbackCommand );
    this->ListSelectedColorButton->AddObserver (vtkKWChangeColorButton::ColorChangedEvent, (vtkCommand *)this->GUICallbackCommand );
    this->ListSymbolScale->AddObserver (vtkKWScale::ScaleValueChangedEvent, (vtkCommand *)this->GUICallbackCommand);
    this->ListTextScale->AddObserver (vtkKWScale::ScaleValueChangedEvent, (vtkCommand *)this->GUICallbackCommand);
    this->ListOpacity->AddObserver (vtkKWScale::ScaleValueChangedEvent, (vtkCommand *)this->GUICallbackCommand);
    
    this->AddObserver(vtkSlicerFiducialsGUI::FiducialListIDModifiedEvent, (vtkCommand *)this->GUICallbackCommand);
}



//---------------------------------------------------------------------------
void vtkSlicerFiducialsGUI::ProcessGUIEvents ( vtkObject *caller,
                                            unsigned long event, void *callData )
{
    // process fiducial list node selector events
    vtkSlicerNodeSelectorWidget *fidListSelector = 
        vtkSlicerNodeSelectorWidget::SafeDownCast(caller);
    if (fidListSelector == this->FiducialListSelectorWidget &&
        event == vtkSlicerNodeSelectorWidget::NodeSelectedEvent )
    {
        vtkDebugMacro("vtkSlicerFiducialsGUI: ProcessGUIEvent Node Selector Event: " << event << ".\n");
        vtkMRMLFiducialListNode *fidList =
            vtkMRMLFiducialListNode::SafeDownCast(this->FiducialListSelectorWidget->GetSelected());
        if (fidList != NULL)
        {
            this->SetFiducialListNode(fidList);
        }
        return;
    }
    vtkMRMLFiducialListNode *fidList =
        vtkMRMLFiducialListNode::SafeDownCast(caller);
    if (fidList == this->MRMLScene->GetNodeByID(this->GetFiducialListNodeID()) &&
        event == vtkCommand::ModifiedEvent)
    {
        return;
    }
    
  // get the currently displayed list
  
  // is there one list?
  vtkMRMLFiducialListNode *activeFiducialListNode = (vtkMRMLFiducialListNode *)this->MRMLScene->GetNodeByID(this->GetFiducialListNodeID());
  if (activeFiducialListNode == NULL)
  {
      std::cerr << "ERROR: No Fiducial List, adding one first!\n";
      this->FiducialListSelectorWidget->ProcessNewNodeCommand("vtkMRMLFiducialListNode",
                                                              this->MRMLScene->GetTagByClassName("vtkMRMLFiducialListNode"));

      // now get the newly active node 
      activeFiducialListNode = (vtkMRMLFiducialListNode *)this->MRMLScene->GetNodeByID(this->GetFiducialListNodeID());
      if (activeFiducialListNode == NULL)
      {
          std::cerr << "ERROR adding a new fiducial list for the point...\n";
          return;
      }
  }
  // save state for undo
  this->MRMLScene->SaveStateForUndo(activeFiducialListNode);

  vtkKWPushButton *button = vtkKWPushButton::SafeDownCast(caller);
  if (button == this->AddFiducialButton  && event ==  vtkKWPushButton::InvokedEvent)
    {
        vtkDebugMacro("vtkSlicerFiducialsGUI: ProcessGUIEvent: Add Fiducial Button event: " << event << ".\n");
       // save state for undo
       this->MRMLScene->SaveStateForUndo();

       // add a fiducial, get the index of the new fiducial
       int modelIndex = activeFiducialListNode->AddFiducial();
       if ( modelIndex < 0 ) 
       {
           // TODO: generate an error...
           std::cerr << "ERROR adding a new fiducial point\n";
           return;
       }
    }
  if (button == this->RemoveFiducialButton && event == vtkKWPushButton::InvokedEvent)
    {
        vtkDebugMacro("vtkSlicerFiducialsGUI: ProcessGUIEvent: Remove Fiducial Button event: " << event << ".\n");
        // check to see if should confirm
        const char * confirmDelete = ((vtkSlicerApplication *)this->GetApplication())->GetConfirmDelete();
        int confirmDeleteFlag = 0;
        if (confirmDelete != NULL &&
            strncmp(confirmDelete, "1", 1) == 0)
        {
            vtkDebugMacro("vtkSlicerFiducialsGUI: ProcessGUIEvent: confirm delete flag is 1\n");
            confirmDeleteFlag = 1;
        }
        else
        {
            vtkDebugMacro("Not confirming deletes, confirmDelete = '" << confirmDelete << "'\n");
        }
        // save state for undo
        this->MRMLScene->SaveStateForUndo();
        
        // get the row that was last selected
        int numRows = this->MultiColumnList->GetWidget()->GetNumberOfSelectedRows();
        if (numRows == 1)
        {
            int row[1];
            this->MultiColumnList->GetWidget()->GetSelectedRows(row);

            if (confirmDeleteFlag)
            {
                // confirm that really want to remove this fiducial
                std::cout << "Removing fiducial " << row[0] << endl;
            }
            
            // then remove that fiducial by index
            activeFiducialListNode->RemoveFiducial(row[0]);
        }
        else
        {
            std::cerr << "Selected rows (" << numRows << ") not 1, just pick one to delete for now\n";
            return;
        }
    }
  if (button == this->RemoveFiducialsButton && event == vtkKWPushButton::InvokedEvent)
    {
        vtkDebugMacro("vtkSlicerFiducialsGUI: ProcessGUIEvent: Remove Fiducials Button event: " << event << ".\n");
        // save state for undo
        this->MRMLScene->SaveStateForUndo();
        activeFiducialListNode->RemoveAllFiducials();
    }
  if (button == this->GetVisibilityToggle()  && event ==  vtkKWPushButton::InvokedEvent)
    {
        vtkDebugMacro("vtkSlicerFiducialsGUI: ProcessGUIEvent: Visibility button event: " << event << ".\n");
       // change the visibility
        activeFiducialListNode->SetVisibility( ! activeFiducialListNode->GetVisibility());
        // update the icon via  process mrml event that should get pushed
        //this->ProcessMRMLEvents(caller, event, callData); 

    }

  // list colour
  vtkKWChangeColorButton *colorButton = vtkKWChangeColorButton::SafeDownCast(caller);
  if (colorButton == this->ListColorButton && event == vtkKWChangeColorButton::ColorChangedEvent)
  {
      vtkDebugMacro("ProcessGUIEvents: list colour button change event\n");
       // change the colour
      activeFiducialListNode->SetColor(this->ListColorButton->GetColor());
        // this->ProcessMRMLEvents(caller, event, callData); 

  }
  if (colorButton == this->ListSelectedColorButton && event == vtkKWChangeColorButton::ColorChangedEvent)
  {
      vtkDebugMacro("ProcessGUIEvents: list selected colour button change event\n");
      // change the selected colour
      activeFiducialListNode->SetSelectedColor(this->ListSelectedColorButton->GetColor());
  }
  
  // list symbol and text sizes
  vtkKWScaleWithEntry *scale = vtkKWScaleWithEntry::SafeDownCast(caller);
  if (scale == this->ListSymbolScale && event == vtkKWScale::ScaleValueChangedEvent)
  {
     activeFiducialListNode->SetSymbolScale(this->ListSymbolScale->GetValue());
      // this->ProcessMRMLEvents(caller, event, callData);                                                     
  }
  else if (scale == this->ListTextScale && event == vtkKWScale::ScaleValueChangedEvent)
  {
      activeFiducialListNode->SetTextScale(this->ListTextScale->GetValue());
      // this->ProcessMRMLEvents(caller, event, callData); 
  }
  else if (scale == this->ListOpacity && event == vtkKWScale::ScaleValueChangedEvent)
  {
      activeFiducialListNode->SetOpacity(this->ListOpacity->GetValue());
  }
  return;
}

//---------------------------------------------------------------------------
void vtkSlicerFiducialsGUI::ProcessLogicEvents ( vtkObject *caller,
                                              unsigned long event, void *callData )
{
    // Fill in
}

//---------------------------------------------------------------------------
void vtkSlicerFiducialsGUI::ProcessMRMLEvents ( vtkObject *caller,
                                             unsigned long event, void *callData )
{    
    vtkDebugMacro("vtkSlicerFiducialsGUI::ProcessMRMLEvents: event = " << event << ".\n");
    
    if (event == vtkCommand::WidgetValueChangedEvent)
    {
        vtkDebugMacro("got a widget value changed event... the list node was changed.\n");
    }

    vtkMRMLFiducialListNode *node = vtkMRMLFiducialListNode::SafeDownCast(caller);
    vtkMRMLFiducialListNode *activeFiducialListNode = (vtkMRMLFiducialListNode *)this->MRMLScene->GetNodeByID(this->GetFiducialListNodeID());
    
    if (node == activeFiducialListNode && event == vtkCommand::ModifiedEvent)
    {
        vtkDebugMacro("\tmodified event on the fiducial list node.\n");
        if (node == NULL)
        {
            vtkDebugMacro("\tBUT: the node is null\n");
            return;
        }
        vtkDebugMacro("\t\tUpdating the GUI\n");
        // update the table
        SetGUIFromList(activeFiducialListNode);
        return;
    }
    if (node == activeFiducialListNode && event == vtkMRMLFiducialListNode::FiducialModifiedEvent)
    {
        vtkDebugMacro("\tfiducial modified event on the active fiducial list.");
        if (node == NULL)
        {
            return;
        }
        SetGUIFromList(activeFiducialListNode);
        return;
    }
    
    if (node == activeFiducialListNode && event == vtkMRMLFiducialListNode::DisplayModifiedEvent)
    {
        vtkDebugMacro("vtkSlicerFiducialsGUI::ProcessMRMLEvents: DisplayModified event on the fiducial list node...\n");
    }
    if (node == vtkMRMLFiducialListNode::SafeDownCast(this->FiducialListSelectorWidget->GetSelected()) && event == vtkCommand::ModifiedEvent)
    {
        vtkDebugMacro("\tmodified event on the fiducial list selected node.\n");
        if (activeFiducialListNode !=  vtkMRMLFiducialListNode::SafeDownCast(this->FiducialListSelectorWidget->GetSelected()))
        {
            // select it first off
            this->SetFiducialListNode(vtkMRMLFiducialListNode::SafeDownCast(this->FiducialListSelectorWidget->GetSelected()));
        }
        SetGUIFromList(activeFiducialListNode);
        return;        
    }    


    vtkDebugMacro("vtkSlicerFiducialsGUI: Done processing mrml events...");
//    std::cerr << "vtkSlicerFiducialsGUI::ProcessMRMLEvents  mismatched
//    caller and event (" << event << ")\n";
}

//---------------------------------------------------------------------------
void vtkSlicerFiducialsGUI::SetGUIFromList(vtkMRMLFiducialListNode * activeFiducialListNode)
{
    if (activeFiducialListNode == NULL)
    {
        return;
    }
    int numPoints = activeFiducialListNode->GetNumberOfFiducials();
    bool deleteFlag = true;
    
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
    // a rows for each point
        
        float *xyz;
        float *wxyz;
        for (int row = 0; row < numPoints; row++)
        {
            // add a row for this point
            if (deleteFlag)
            {
                vtkDebugMacro("SetGUIFromList: Adding point " << row << " to the table" << endl);
                this->MultiColumnList->GetWidget()->AddRow();
            }
            vtkDebugMacro("SetGUIFromList: getting " << row << "th fiducial xyz - total fids = " << numPoints);
            // now populate it
            xyz = activeFiducialListNode->GetNthFiducialXYZ(row);
            if (xyz == NULL)
            {
                std::cerr << "SetGUIFromList: ERROR: got null xyz for point " << row << endl;
            }
            vtkDebugMacro("Getting nth fiducial orientation");            
            wxyz = activeFiducialListNode->GetNthFiducialOrientation(row);
            
            if (activeFiducialListNode->GetNthFiducialLabelText(row) != NULL)
            {
                if (strcmp(this->MultiColumnList->GetWidget()->GetCellText(row,this->NameColumn), activeFiducialListNode->GetNthFiducialLabelText(row)) != 0)
                {
                    this->MultiColumnList->GetWidget()->SetCellText(row,this->NameColumn,activeFiducialListNode->GetNthFiducialLabelText(row));
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
            if (deleteFlag || this->MultiColumnList->GetWidget()->GetCellTextAsInt(row,this->SelectedColumn) != (activeFiducialListNode->GetNthFiducialSelected(row) ? 1 : 0))
            {
                this->MultiColumnList->GetWidget()->SetCellTextAsInt(row,this->SelectedColumn,(activeFiducialListNode->GetNthFiducialSelected(row) ? 1 : 0));
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
            if (wxyz != NULL)
            {
                if (deleteFlag || this->MultiColumnList->GetWidget()->GetCellTextAsDouble(row,this->OrWColumn) != wxyz[0])
                {
                    this->MultiColumnList->GetWidget()->SetCellTextAsDouble(row,this->OrWColumn,wxyz[0]);
                }
                if (deleteFlag || this->MultiColumnList->GetWidget()->GetCellTextAsDouble(row,this->OrXColumn) != wxyz[1])
                {
                    this->MultiColumnList->GetWidget()->SetCellTextAsDouble(row,this->OrXColumn,wxyz[1]);
                }
                if (deleteFlag || this->MultiColumnList->GetWidget()->GetCellTextAsDouble(row,this->OrYColumn) != wxyz[2])
                {
                    this->MultiColumnList->GetWidget()->SetCellTextAsDouble(row,this->OrYColumn,wxyz[2]);
                }
                if (deleteFlag || this->MultiColumnList->GetWidget()->GetCellTextAsDouble(row,this->OrZColumn) != wxyz[3])
                {
                    this->MultiColumnList->GetWidget()->SetCellTextAsDouble(row,this->OrZColumn,wxyz[3]);
                }
            }
        }
        
        vtkDebugMacro("Now going to update GUI from the logic's active list");
    // update the visibility, color, scale buttons to match the displayed list's
    if (activeFiducialListNode == NULL)
    {
        std::cerr << "vtkSlicerFiducialsGUI::SetGUIFromList: ERROR: no active fiducial list node in the gui class!\n";                return;
    }
    vtkDebugMacro(<< "\tupdating the visibility button\n");
    if (this->GetVisibilityToggle() != NULL &&
        this->GetVisibilityIcons() != NULL)
    {
        if (activeFiducialListNode->GetVisibility() > 0)
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
        std::cerr << "ERROR; trying up update null visibility toggle!\n";
    }
    
    // color
    vtkDebugMacro(<< "\tupdating the colour\n");
    double *nodeColor = activeFiducialListNode->GetColor();
    if (this->ListColorButton != NULL)
    {
        double *buttonColor = this->ListColorButton->GetColor();
        if (nodeColor != NULL && buttonColor != NULL && 
            (nodeColor[0] != buttonColor[0] ||
             nodeColor[1] != buttonColor[1] ||
             nodeColor[2] != buttonColor[2]))
        {
            vtkDebugMacro("Updating list color button\n");
            this->ListColorButton->SetColor(nodeColor);
        }
    }
    else
    {
        std::cerr << "No colour button!\n";
    }

    // selected color
    vtkDebugMacro(<< "\tupdating the selected colour\n");
    double *nodeSelectedColor = activeFiducialListNode->GetSelectedColor();
    if (this->ListSelectedColorButton != NULL)
    {
        double *buttonSelectedColor = this->ListSelectedColorButton->GetColor();
        if (nodeSelectedColor != NULL && buttonSelectedColor != NULL && 
            (nodeSelectedColor[0] != buttonSelectedColor[0] ||
             nodeSelectedColor[1] != buttonSelectedColor[1] ||
             nodeSelectedColor[2] != buttonSelectedColor[2]))
        {
            vtkDebugMacro("Updating list selected color button\n");
            this->ListSelectedColorButton->SetColor(nodeSelectedColor);
        }
    }
    else
    {
        std::cerr << "No selected colour button!\n";
    }
    
    // symbol scale
    vtkDebugMacro(<< "\tupdating the symbol scale");
    double scale = activeFiducialListNode->GetSymbolScale();
    if (this->ListSymbolScale != NULL &&
        scale != this->ListSymbolScale->GetValue())
    {
        this->ListSymbolScale->SetValue(scale);
    }
    

    // text scale
    vtkDebugMacro(<< "\tupdating the text scale.");
    scale = activeFiducialListNode->GetTextScale();
    if (this->ListTextScale != NULL &&
        scale != this->ListTextScale->GetValue())
    {
        this->ListTextScale->SetValue(scale);
    }

    // opacity
    vtkDebugMacro(<< "\tupdating the opacity");
    scale = activeFiducialListNode->GetOpacity();
    if (this->ListOpacity != NULL &&
        scale != this->ListOpacity->GetValue())
    {
        this->ListOpacity->SetValue(scale);
    }
}
//---------------------------------------------------------------------------
void vtkSlicerFiducialsGUI::Enter ( )
{
    std::cout << "vtkSlicerFiducialsGUI: Enter\n";
}

//---------------------------------------------------------------------------
void vtkSlicerFiducialsGUI::Exit ( )
{
    std::cout <<  "vtkSlicerFiducialsGUI: Exit\n";
}




//---------------------------------------------------------------------------
void vtkSlicerFiducialsGUI::BuildGUI ( )
{
    vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
  // Define your help text here.
  const char *help = "**Fiducials Module:** Creates and manages lists of Fiducial points. ";

    // ---
    // MODULE GUI FRAME 
    // configure a page for a model loading UI for now.
    // later, switch on the modulesButton in the SlicerControlGUI
    // ---
    // create a page
    this->UIPanel->AddPage ( "Fiducials", "Fiducials", NULL );

    vtkKWWidget *page = this->UIPanel->GetPageWidget ( "Fiducials" );
    
    // HELP FRAME
    vtkSlicerModuleCollapsibleFrame *modHelpFrame = vtkSlicerModuleCollapsibleFrame::New ( );
    modHelpFrame->SetParent (page );
    modHelpFrame->Create ( );
    modHelpFrame->CollapseFrame ( );
    modHelpFrame->SetLabelText ("Help");
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                  modHelpFrame->GetWidgetName(), this->UIPanel->GetPageWidget("Fiducials")->GetWidgetName());

    // configure the parent classes help text widget
    this->HelpText->SetParent ( modHelpFrame->GetFrame() );
    this->HelpText->Create ( );
    this->HelpText->SetText ( help );
    this->HelpText->SetReliefToFlat ( );
    this->HelpText->SetWrapToWord ( );
    this->HelpText->ReadOnlyOn ( );
    this->HelpText->QuickFormattingOn ( );
    this->HelpText->SetBalloonHelpString ( "" );
    app->Script ( "pack %s -side top -fill x -expand y -anchor w -padx 2 -pady 4",
                  this->HelpText->GetWidgetName ( ) );

    // ---
    // DISPLAY FRAME            
    vtkSlicerModuleCollapsibleFrame *displayFrame = vtkSlicerModuleCollapsibleFrame::New ( );
    displayFrame->SetParent ( page );
    displayFrame->Create ( );
    displayFrame->SetLabelText ("Display");
    displayFrame->ExpandFrame ( );
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                  displayFrame->GetWidgetName(),
                  this->UIPanel->GetPageWidget("Fiducials")->GetWidgetName());
  
    // node selector
    this->FiducialListSelectorWidget = vtkSlicerNodeSelectorWidget::New() ;
    this->FiducialListSelectorWidget->SetParent(displayFrame->GetFrame());
    this->FiducialListSelectorWidget->Create();
    this->FiducialListSelectorWidget->SetNodeClass("vtkMRMLFiducialListNode", NULL, NULL, NULL);
    this->FiducialListSelectorWidget->NewNodeEnabledOn();
    this->FiducialListSelectorWidget->SetMRMLScene(this->GetMRMLScene());
    this->FiducialListSelectorWidget->SetBorderWidth(2);
    this->FiducialListSelectorWidget->SetPadX(2);
    this->FiducialListSelectorWidget->SetPadY(2);
    //this->FiducialListSelectorWidget->GetWidget()->IndicatorVisibilityOff();
    this->FiducialListSelectorWidget->GetWidget()->SetWidth(24);
    this->FiducialListSelectorWidget->SetLabelText( "Fiducial List Select: ");
    this->FiducialListSelectorWidget->SetBalloonHelpString("Select a fiducial list from the current mrml scene.");
    this->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
                  this->FiducialListSelectorWidget->GetWidgetName());
    
    // scale frame
    vtkKWFrame *scaleFrame = vtkKWFrame::New();
    scaleFrame->SetParent ( displayFrame->GetFrame() );
    scaleFrame->Create ( );
    app->Script ("pack %s -side top -anchor nw -fill x -pady 0 -in %s",
                 scaleFrame->GetWidgetName(),
                 displayFrame->GetFrame()->GetWidgetName());
    
    // text scale
    this->ListTextScale = vtkKWScaleWithEntry::New();
    this->ListTextScale->SetParent( scaleFrame );
    this->ListTextScale->Create();
    this->ListTextScale->SetLabelText("Text Scale:");
    this->ListTextScale->SetBalloonHelpString ( "Set the scale of the fiducial list points text.");
    this->ListTextScale->GetWidget()->SetRange(0.0, 20.0);
    this->ListTextScale->GetWidget()->SetOrientationToHorizontal ();
    this->ListTextScale->GetWidget()->SetResolution(0.5);
    this->ListTextScale->SetEntryWidth(5);
    //app->Script("pack %s -side top -anchor w -padx 2 -pady 2", 
    //            this->ListTextScale->GetWidgetName());

    // symbol scale
    this->ListSymbolScale = vtkKWScaleWithEntry::New();
    this->ListSymbolScale->SetParent( scaleFrame );
    this->ListSymbolScale->Create();
    this->ListSymbolScale->SetLabelText("Symbol Scale:");
    this->ListSymbolScale->SetBalloonHelpString ( "Set the scale of the fiducial list symbols.");
    this->ListSymbolScale->GetWidget()->SetRange(0.0, 80.0);
    this->ListSymbolScale->GetWidget()->SetOrientationToHorizontal();
    this->ListSymbolScale->GetWidget()->SetResolution(0.5);
    this->ListSymbolScale->SetEntryWidth(5);

    app->Script("pack %s %s -side left -anchor w -padx 2 -pady 2 -in %s", 
                this->ListTextScale->GetWidgetName(), this->ListSymbolScale->GetWidgetName(),
                scaleFrame->GetWidgetName());

        // visibility and opacity frame
    vtkKWFrame *visibilityOpacityFrame = vtkKWFrame::New();
    visibilityOpacityFrame->SetParent ( displayFrame->GetFrame() );
    visibilityOpacityFrame->Create ( );
    app->Script ("pack %s -side top -anchor nw -fill x -pady 0 -in %s",
                 visibilityOpacityFrame->GetWidgetName(),
                 displayFrame->GetFrame()->GetWidgetName());
    
    // visibility
    this->VisibilityIcons = vtkSlicerVisibilityIcons::New ( );
    this->VisibilityToggle = vtkKWPushButton::New();
    this->VisibilityToggle->SetParent ( visibilityOpacityFrame );
    this->VisibilityToggle->Create ( );
    this->VisibilityToggle->SetReliefToFlat ( );
    this->VisibilityToggle->SetOverReliefToNone ( );
    this->VisibilityToggle->SetBorderWidth ( 0 );
    this->VisibilityToggle->SetImageToIcon ( this->VisibilityIcons->GetInvisibleIcon ( ) );        
    this->VisibilityToggle->SetBalloonHelpString ( "Toggles fiducial list visibility in the MainViewer." );
    this->VisibilityToggle->SetText ("Visibility");
    
    // opacity
    this->ListOpacity = vtkKWScaleWithEntry::New();
    this->ListOpacity->SetParent( visibilityOpacityFrame );
    this->ListOpacity->Create();
    this->ListOpacity->SetLabelText("Opacity:");
    this->ListOpacity->SetBalloonHelpString ( "Set the opacity of the fiducial list symbols.");
    this->ListOpacity->GetWidget()->SetRange(0.0, 1.0);
    this->ListOpacity->GetWidget()->SetOrientationToHorizontal();
    this->ListOpacity->GetWidget()->SetResolution(0.1);
    this->ListOpacity->SetEntryWidth(5);
    app->Script("pack %s %s -side left -anchor w -padx 2 -pady 2 -in %s", 
                this->VisibilityToggle->GetWidgetName(), this->ListOpacity->GetWidgetName(),
                visibilityOpacityFrame->GetWidgetName());

    // colour frame
    vtkKWFrame *colourFrame = vtkKWFrame::New();
    colourFrame->SetParent ( displayFrame->GetFrame() );
    colourFrame->Create ( );
    app->Script ("pack %s -side top -anchor nw -fill x -pady 0 -in %s",
                 colourFrame->GetWidgetName(),
                 displayFrame->GetFrame()->GetWidgetName());
    
    // color
    this->ListColorButton = vtkKWChangeColorButton::New();
    this->ListColorButton->SetParent( colourFrame );
    this->ListColorButton->Create();
    this->ListColorButton->SetBorderWidth(0);
    this->ListColorButton->SetBalloonHelpString("Change the colour of the fiducial list symbols and text in the MainViewer");
    this->ListColorButton->SetDialogTitle("List symbol and text color");
    this->ListColorButton->SetLabelText("Set Color");

    // selected colour
    this->ListSelectedColorButton = vtkKWChangeColorButton::New();
    this->ListSelectedColorButton->SetParent( colourFrame );
    this->ListSelectedColorButton->Create();
    this->ListSelectedColorButton->SetBorderWidth(0);
    this->ListSelectedColorButton->SetBalloonHelpString("Change the colour of the selected fiducial list symbols and text in the MainViewer");
    
    this->ListSelectedColorButton->SetDialogTitle("List selected symbol and text color");
    this->ListSelectedColorButton->SetLabelText("Set Selected Color");

    // pack the colours 
    app->Script("pack %s %s -side left -anchor w -padx 4 -pady 2 -in %s",
                this->ListColorButton->GetWidgetName(), this->ListSelectedColorButton->GetWidgetName(),
                colourFrame->GetWidgetName());
    
    // ---
    // LIST FRAME
    
    vtkSlicerModuleCollapsibleFrame *listFrame = vtkSlicerModuleCollapsibleFrame::New();
    listFrame->SetParent( page );
    listFrame->Create();
    listFrame->SetLabelText("Fiducial List");
    listFrame->ExpandFrame();
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                  listFrame->GetWidgetName(), this->UIPanel->GetPageWidget("Fiducials")->GetWidgetName());


    
    // add the multicolumn list to show the points
    this->MultiColumnList = vtkKWMultiColumnListWithScrollbars::New ( );
    this->MultiColumnList->SetParent ( listFrame->GetFrame() );
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
    this->MultiColumnList->GetWidget()->AddColumn("OrW");
    this->MultiColumnList->GetWidget()->AddColumn("OrX");
    this->MultiColumnList->GetWidget()->AddColumn("OrY");
    this->MultiColumnList->GetWidget()->AddColumn("OrZ");
    
    // make the selected column editable by checkbox
    this->MultiColumnList->GetWidget()->SetColumnEditWindowToCheckButton(this->SelectedColumn);
    
    // now set the attributes that are equal across the columns
    int col;
    for (col = 0; col < this->NumberOfColumns; col++)
    {        
        this->MultiColumnList->GetWidget()->SetColumnWidth(col, 6);
        this->MultiColumnList->GetWidget()->SetColumnAlignmentToLeft(col);
        this->MultiColumnList->GetWidget()->ColumnEditableOn(col);
    }
    // set the name column width to be higher
    this->MultiColumnList->GetWidget()->SetColumnWidth(this->NameColumn, 15);
    // set the selected column width a bit higher
    this->MultiColumnList->GetWidget()->SetColumnWidth(this->SelectedColumn, 9);
    
    app->Script ( "pack %s -fill both -expand true",
                  this->MultiColumnList->GetWidgetName());
//                  listFrame->GetWidgetName());
    this->MultiColumnList->GetWidget()->SetCellUpdatedCommand(this, "UpdateElement");

    // button frame
    vtkKWFrame *buttonFrame = vtkKWFrame::New();
    buttonFrame->SetParent ( listFrame->GetFrame() );
    buttonFrame->Create ( );
    app->Script ("pack %s -side top -anchor nw -fill x -pady 0 -in %s",
                 buttonFrame->GetWidgetName(),
                 listFrame->GetFrame()->GetWidgetName());
    
     // add an add fiducial button
    this->AddFiducialButton = vtkKWPushButton::New ( );
    this->AddFiducialButton->SetParent ( buttonFrame );
    this->AddFiducialButton->Create ( );
    this->AddFiducialButton->SetText ("Add Fiducial");
    this->AddFiducialButton->SetBalloonHelpString("Add a fiducial point to the current list");
    
    // add a remove fiducial button
    this->RemoveFiducialButton = vtkKWPushButton::New ( );
    this->RemoveFiducialButton->SetParent ( buttonFrame );
    this->RemoveFiducialButton->Create ( );
    this->RemoveFiducialButton->SetText ("Remove Fiducial");
    this->RemoveFiducialButton->SetBalloonHelpString("Remove the last fiducial that was clicked on in the table from the list.");


    // add a remove all fiducials from this list button
    this->RemoveFiducialsButton = vtkKWPushButton::New ( );
    this->RemoveFiducialsButton->SetParent ( buttonFrame );
    this->RemoveFiducialsButton->Create ( );
    this->RemoveFiducialsButton->SetText ("Remove All Fiducials");
    this->RemoveFiducialsButton->SetBalloonHelpString("Remove all fiducial points from the list.");
  
    app->Script("pack %s %s %s -side left -anchor w -padx 4 -pady 2", 
                this->AddFiducialButton->GetWidgetName(),
                this->RemoveFiducialButton->GetWidgetName(),
                this->RemoveFiducialsButton->GetWidgetName());

    if (this->MRMLScene != NULL)
    {
        if (this->MRMLScene->HasObserver(vtkCommand::ModifiedEvent, (vtkCommand *)this->MRMLCallbackCommand ) == 0)
        {
            this->MRMLScene->AddObserver(vtkCommand::ModifiedEvent, (vtkCommand *)this->MRMLCallbackCommand );
        }
        // add in observers on node added/removed
    }
     
    // deleting frame widgets
    buttonFrame->Delete ();
    colourFrame->Delete ();
    visibilityOpacityFrame->Delete ();
    scaleFrame->Delete();    
    displayFrame->Delete ( );
    listFrame->Delete();
    modHelpFrame->Delete ( );
}

//---------------------------------------------------------------------------
void vtkSlicerFiducialsGUI::UpdateElement(int row, int col, char * str)
{
    vtkDebugMacro("row = " << row << ", col = " << col << ", str = " << str << "\n");

    // make sure that the row and column exists in the table
    if ((row >= 0) && (row < this->MultiColumnList->GetWidget()->GetNumberOfRows()) &&
        (col >= 0) && (col < this->MultiColumnList->GetWidget()->GetNumberOfColumns()))
        {
            vtkMRMLFiducialListNode *activeFiducialListNode = (vtkMRMLFiducialListNode *)this->MRMLScene->GetNodeByID(this->GetFiducialListNodeID());
            // is there an active list?
            if (activeFiducialListNode == NULL)
            {
                // 
                std::cerr << "UpdateElement: ERROR: No Fiducial List, add one first!\n";
                return;
            }
        
            // now update the requested value
            if (col == this->NameColumn)
            {
                activeFiducialListNode->SetNthFiducialLabelText(row, str);
            }
            else if (col == this->SelectedColumn)
            {
                // selected
                vtkDebugMacro("UpdateElement: setting node " <<  activeFiducialListNode->GetNthFiducialLabelText(row) << "'s selected flag to " << str << endl);
                activeFiducialListNode->SetNthFiducialSelected(row, (atoi(str) == 1));
            }
            else if (col >= this->XColumn && col <= this->ZColumn)
            {
                // get the current xyz
                float * xyz = activeFiducialListNode->GetNthFiducialXYZ(row);
                // now set the new one
                float newCoordinate = atof(str);
                if (col == this->XColumn) { activeFiducialListNode->SetNthFiducialXYZ(row, newCoordinate, xyz[1], xyz[2]); }
                if (col == this->YColumn) { activeFiducialListNode->SetNthFiducialXYZ(row, xyz[0], newCoordinate, xyz[2]); }
                if (col == this->ZColumn) { activeFiducialListNode->SetNthFiducialXYZ(row, xyz[0], xyz[1], newCoordinate); }
            }
            else if (col >= this->OrWColumn  && col <= this->OrZColumn)
            {
                float * wxyz = activeFiducialListNode->GetNthFiducialOrientation(row);
                float newCoordinate = atof(str);
                if (col == this->OrWColumn) { activeFiducialListNode->SetNthFiducialOrientation(row, newCoordinate, wxyz[1], wxyz[2], wxyz[3]); }
                if (col == this->OrXColumn) { activeFiducialListNode->SetNthFiducialOrientation(row, wxyz[0], newCoordinate, wxyz[2], wxyz[3]); }
                if (col == this->OrYColumn) { activeFiducialListNode->SetNthFiducialOrientation(row, wxyz[0], wxyz[1], newCoordinate, wxyz[3]); }
                if (col == this->OrZColumn) { activeFiducialListNode->SetNthFiducialOrientation(row, wxyz[0], wxyz[1], wxyz[2], newCoordinate); }
            }
            else
            {
                std::cerr << "UpdateElement: ERROR: invalid column number " << col << ", valid values are 0-" << this->NumberOfColumns << endl;
                return;
            }
        }
    else
    {
        std::cerr << "Invalid row " << row << " or column " << col <<  ", valid columns are 0-" << this->NumberOfColumns << "\n";
    }
}

//---------------------------------------------------------------------------
void vtkSlicerFiducialsGUI::SetFiducialListNode (vtkMRMLFiducialListNode *fiducialListNode)
{
    if (fiducialListNode == NULL)
    {
        std::cerr << "ERROR: SetFiducialListNode - list node is null.\n";
        return;
    }
    // save the ID
    vtkDebugMacro("setting the fid list node id to " << fiducialListNode->GetID());
    this->SetFiducialListNodeID(fiducialListNode->GetID());
}

//---------------------------------------------------------------------------
void vtkSlicerFiducialsGUI::SetFiducialListNodeID (char * id)
{
    if (this->GetFiducialListNodeID() != NULL &&
        id != NULL &&
        strcmp(id,this->GetFiducialListNodeID()) == 0)
    {
        vtkDebugMacro("no change in id, not doing anything for now: " << id << endl);
        return;
    }

    // get the old node
    vtkMRMLFiducialListNode *fidlist = vtkMRMLFiducialListNode::SafeDownCast(this->MRMLScene->GetNodeByID(this->GetFiducialListNodeID()));
    if (fidlist != NULL)
    {
        // remove observers on old node
        fidlist->RemoveObservers(vtkMRMLFiducialListNode::DisplayModifiedEvent, (vtkCommand *)this->MRMLCallbackCommand);
        fidlist->RemoveObservers(vtkCommand::ModifiedEvent, (vtkCommand *)this->MRMLCallbackCommand);
        fidlist->RemoveObservers(vtkMRMLFiducialListNode::FiducialModifiedEvent, (vtkCommand *)this->MRMLCallbackCommand);
    }
    else
    {
        // probably wasn't set yet
    }
       
    // set the id properly - see the vtkSetStringMacro
    this->FiducialListNodeID = id;

    if (id == NULL)
    {
        std::cerr << "SetFiducialListNodeID: NULL input id, removed observers and returning.\n";
        return;
    }
    
    // get the new node
    fidlist = vtkMRMLFiducialListNode::SafeDownCast(this->MRMLScene->GetNodeByID(this->GetFiducialListNodeID()));
    // set up observers on the new node
    if (fidlist != NULL)
    {
        if (fidlist->HasObserver(vtkMRMLFiducialListNode::DisplayModifiedEvent, (vtkCommand *)this->MRMLCallbackCommand) == 0)
        {
            fidlist->AddObserver(vtkMRMLFiducialListNode::DisplayModifiedEvent, (vtkCommand *)this->MRMLCallbackCommand);
        }
        if (fidlist->HasObserver(vtkCommand::ModifiedEvent, (vtkCommand *)this->MRMLCallbackCommand) == 0)
        {
            fidlist->AddObserver(vtkCommand::ModifiedEvent, (vtkCommand *)this->MRMLCallbackCommand);
        }
        if (fidlist->HasObserver(vtkMRMLFiducialListNode::FiducialModifiedEvent, (vtkCommand *)this->MRMLCallbackCommand) == 0)
        {
            fidlist->AddObserver(vtkMRMLFiducialListNode::FiducialModifiedEvent, (vtkCommand *)this->MRMLCallbackCommand);
        }
        // set up the GUI
        this->SetGUIFromList(fidlist);
    }
    else
    {
        std::cerr << "ERROR: unable to get the mrml fiducial node to observe!\n";
    }
}
