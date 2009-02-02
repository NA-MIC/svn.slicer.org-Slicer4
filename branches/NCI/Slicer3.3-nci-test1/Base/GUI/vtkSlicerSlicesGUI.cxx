#include "vtkObject.h"
#include "vtkObjectFactory.h"
#include "vtkCommand.h"

#include "vtkSlicerSlicesGUI.h"
//#include "vtkSlicerSliceGUICollection.h"
#include "vtkSlicerSliceGUI.h"
#include "vtkSlicerSliceLogic.h"
#include "vtkMRMLSliceNode.h"
#include "vtkSlicerApplication.h"
#include "vtkSlicerApplicationLogic.h"

#include "vtkKWApplication.h"
#include "vtkKWFrame.h"
#include "vtkKWCheckButton.h"
#include "vtkKWCheckButtonWithLabel.h"
#include "vtkKWLabel.h"
#include "vtkKWLabelWithLabel.h"
#include "vtkKWRadioButton.h"
#include "vtkKWRadioButtonSet.h"
#include "vtkKWRadioButtonSetWithLabel.h"
#include "vtkKWEntry.h"
#include "vtkKWEntryWithLabel.h"
#include "vtkKWFrame.h"
#include "vtkKWMenu.h"
#include "vtkKWMenuButton.h"
#include "vtkKWMenuButtonWithLabel.h"
#include "vtkKWPushButton.h"
#include "vtkKWPushButtonWithLabel.h"

#include "vtkSlicerModuleCollapsibleFrame.h"
#include "vtkSlicerNodeSelectorWidget.h"
#include "vtkSlicerVisibilityIcons.h"

#include <map>

// Private implementaton of an std::map
class SliceGUIMap : public std::map<std::string, vtkSmartPointer<vtkSlicerSliceGUI> > {};
class ParameterWidgetMap : public std::map<std::string, vtkSmartPointer<vtkKWCoreWidget> > {};


//---------------------------------------------------------------------------
vtkStandardNewMacro (vtkSlicerSlicesGUI);
vtkCxxRevisionMacro(vtkSlicerSlicesGUI, "$Revision: 1.0 $");


//---------------------------------------------------------------------------
vtkSlicerSlicesGUI::vtkSlicerSlicesGUI (  )
{
  this->InternalSliceGUIMap = new SliceGUIMap;
  this->InternalParameterWidgetMap = new ParameterWidgetMap;

  this->VisibilityIcons = 0;
  this->SliceNode = 0;
}


//---------------------------------------------------------------------------
vtkSlicerSlicesGUI::~vtkSlicerSlicesGUI ( )
{
  if (this->InternalSliceGUIMap)
    {
    delete this->InternalSliceGUIMap;
    }
  if (this->InternalParameterWidgetMap)
    {
    delete this->InternalParameterWidgetMap;
    }

  if (this->VisibilityIcons)
    {
    this->VisibilityIcons->Delete();
    }
}

void vtkSlicerSlicesGUI::AddSliceGUI(const char *layoutName, vtkSlicerSliceGUI *pSliceGUI)
{
  std::string sMRMLNodeLayoutName = layoutName;
  (*this->InternalSliceGUIMap)[sMRMLNodeLayoutName] = pSliceGUI;
}

//---------------------------------------------------------------------------
void vtkSlicerSlicesGUI::AddAndObserveSliceGUI ( const char *layoutName, vtkSlicerSliceGUI *pSliceGUI )
{
  this->AddSliceGUI ( layoutName, pSliceGUI );
  pSliceGUI->AddGUIObservers ( );
}

vtkSlicerSliceGUI* vtkSlicerSlicesGUI::GetSliceGUI(const char *layoutName)
{
  if (this->InternalSliceGUIMap)
    {
    SliceGUIMap::const_iterator gend = (*this->InternalSliceGUIMap).end();
    SliceGUIMap::const_iterator git = (*this->InternalSliceGUIMap).find(layoutName);
    
    if ( git != gend)
      return (vtkSlicerSliceGUI::SafeDownCast((*git).second));
    else
      return NULL;
    }
  else
    return NULL;
}

//---------------------------------------------------------------------------
void vtkSlicerSlicesGUI::AddGUIObservers ( )
{
  // TODO: add observers for all widgets in slicesGUIs UIpanel.


  (*this->InternalParameterWidgetMap)["SliceNodeSelector"]->AddObserver (vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand );  

  vtkKWMenuButtonWithLabel::SafeDownCast((*this->InternalParameterWidgetMap)["Orientation"])->GetWidget()->GetMenu()->AddObserver (vtkKWMenu::MenuItemInvokedEvent, (vtkCommand *)this->GUICallbackCommand );  

  vtkKWPushButtonWithLabel::SafeDownCast((*this->InternalParameterWidgetMap)["SliceVisible"])->GetWidget()->AddObserver (vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );  

  vtkKWCheckButtonWithLabel::SafeDownCast((*this->InternalParameterWidgetMap)["WidgetVisible"])->GetWidget()->AddObserver (vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );  

  (*this->InternalParameterWidgetMap)["LayoutRows"]->AddObserver (vtkKWEntry::EntryValueChangedEvent, (vtkCommand *)this->GUICallbackCommand );  

  (*this->InternalParameterWidgetMap)["LayoutColumns"]->AddObserver (vtkKWEntry::EntryValueChangedEvent, (vtkCommand *)this->GUICallbackCommand );  

  vtkKWRadioButtonSetWithLabel *rbs = vtkKWRadioButtonSetWithLabel::SafeDownCast((*this->InternalParameterWidgetMap)["SliceSpacingMode"]);
  if (rbs)
    {
    int num = rbs->GetWidget()->GetNumberOfWidgets();
    for (int i=0; i < num; ++i)
      {
      int id = rbs->GetWidget()->GetIdOfNthWidget(i);
      vtkKWRadioButton* rb = rbs->GetWidget()->GetWidget(id);
      rb->AddObserver(vtkKWRadioButton::SelectedStateChangedEvent,
                      (vtkCommand *) this->GUICallbackCommand);
      }
    }

  vtkKWEntryWithLabel::SafeDownCast((*this->InternalParameterWidgetMap)["PrescribedSpacing"])->GetWidget()->AddObserver (vtkKWEntry::EntryValueChangedEvent, (vtkCommand *)this->GUICallbackCommand );

  if (this->SliceNode)
    {
    this->SliceNode->AddObserver(vtkCommand::ModifiedEvent, (vtkCommand *) this->MRMLCallbackCommand);
    }
  
}


//---------------------------------------------------------------------------
void vtkSlicerSlicesGUI::RemoveGUIObservers ( )
{
  SliceGUIMap::const_iterator git;
  for (git = this->InternalSliceGUIMap->begin();
       git != this->InternalSliceGUIMap->end(); ++git)
    {
    vtkSlicerSliceGUI *g = vtkSlicerSliceGUI::SafeDownCast((*git).second);
    g->RemoveGUIObservers();
    }

  ParameterWidgetMap::const_iterator sit;
  ParameterWidgetMap::const_iterator send;
  send = (*this->InternalParameterWidgetMap).end();
        
  sit = (*this->InternalParameterWidgetMap).find("SliceNodeSelector");
  if ( sit != send)
    {
    (*sit).second->RemoveObservers ( vtkSlicerNodeSelectorWidget::NodeSelectedEvent,  (vtkCommand *)this->GUICallbackCommand );
    }

  sit = (*this->InternalParameterWidgetMap).find("SliceNodeSelector");
  if ( sit != send)
    {
    vtkKWMenuButtonWithLabel::SafeDownCast((*sit).second)->GetWidget()->GetMenu()->RemoveObservers ( vtkKWMenu::MenuItemInvokedEvent,  (vtkCommand *)this->GUICallbackCommand );
    }
        

  sit = (*this->InternalParameterWidgetMap).find("SliceVisible");
  if ( sit != send)
    {
    vtkKWPushButtonWithLabel::SafeDownCast((*sit).second)->GetWidget()->RemoveObservers ( vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );
    }

  sit = (*this->InternalParameterWidgetMap).find("WidgetVisible");
  if ( sit != send)
    {
    vtkKWCheckButtonWithLabel::SafeDownCast((*sit).second)->GetWidget()->RemoveObservers ( vtkKWCheckButton::SelectedStateChangedEvent,  (vtkCommand *)this->GUICallbackCommand );
    }

  sit = (*this->InternalParameterWidgetMap).find("LayoutRows");
  if ( sit != send)
    {
    (*sit).second->RemoveObservers ( vtkKWEntry::EntryValueChangedEvent,  (vtkCommand *)this->GUICallbackCommand );
    }

  sit = (*this->InternalParameterWidgetMap).find("LayoutColumns");
  if ( sit != send)
    {
    (*sit).second->RemoveObservers ( vtkKWEntry::EntryValueChangedEvent,  (vtkCommand *)this->GUICallbackCommand );
    }

  sit = (*this->InternalParameterWidgetMap).find("SliceSpacingMode");
  vtkKWRadioButtonSetWithLabel *rbs = vtkKWRadioButtonSetWithLabel::SafeDownCast((*sit).second);
  if (rbs)
    {
    int num = rbs->GetWidget()->GetNumberOfWidgets();
    for (int i=0; i < num; ++i)
      {
      int id = rbs->GetWidget()->GetIdOfNthWidget(i);
      vtkKWRadioButton* rb = rbs->GetWidget()->GetWidget(id);
      rb->RemoveObservers(vtkKWRadioButton::SelectedStateChangedEvent,
                          (vtkCommand *) this->GUICallbackCommand);
      }
    }
        
  sit = (*this->InternalParameterWidgetMap).find("PrescribedSpacing");
  if ( sit != send)
    {
    vtkKWEntryWithLabel::SafeDownCast((*sit).second)->GetWidget()->RemoveObservers ( vtkKWEntry::EntryValueChangedEvent,  (vtkCommand *)this->GUICallbackCommand );
    }

  if (this->SliceNode)
    {
    this->SliceNode->RemoveObservers(vtkCommand::ModifiedEvent, (vtkCommand *) this->MRMLCallbackCommand);
    }
}

void vtkSlicerSlicesGUI::RemoveGUIMapObservers()
{
  SliceGUIMap::const_iterator git;
  for (git = this->InternalSliceGUIMap->begin();
       git != this->InternalSliceGUIMap->end(); ++git)
    {
    vtkSlicerSliceGUI *g = vtkSlicerSliceGUI::SafeDownCast((*git).second);
    g->RemoveGUIObservers();
    }
}

vtkSlicerSliceGUI* vtkSlicerSlicesGUI::GetFirstSliceGUI ()
{
  if (this->InternalSliceGUIMap)
    {
    SliceGUIMap::const_iterator git;
    git = this->InternalSliceGUIMap->begin();
    return (vtkSlicerSliceGUI::SafeDownCast((*git).second));
    }
  else
    return NULL;
}

const char* vtkSlicerSlicesGUI::GetFirstSliceGUILayoutName()
{
  if (this->InternalSliceGUIMap)
    {
    SliceGUIMap::const_iterator git;
    git = this->InternalSliceGUIMap->begin();
    return ((*git).first.c_str());
    }
  else
    return NULL;
}

int vtkSlicerSlicesGUI::GetNumberOfSliceGUI()
{ 
  if (this->InternalSliceGUIMap)
    return (int)(this->InternalSliceGUIMap->size());
  else
    return 0;
}

vtkSlicerSliceGUI* vtkSlicerSlicesGUI::GetNextSliceGUI(const char *layoutName)
{
  if (this->InternalSliceGUIMap)
    {
    SliceGUIMap::const_iterator gend = (*this->InternalSliceGUIMap).end();
    SliceGUIMap::const_iterator git = (*this->InternalSliceGUIMap).find(layoutName);
    git++;

    if ( git != gend)
      return (vtkSlicerSliceGUI::SafeDownCast((*git).second));
    else
      return NULL;
    }
  else
    return NULL;
}

const char* vtkSlicerSlicesGUI::GetNextSliceGUILayoutName(const char *layoutName)
{
  if (this->InternalSliceGUIMap)
    {
    SliceGUIMap::const_iterator gend = (*this->InternalSliceGUIMap).end();
    SliceGUIMap::const_iterator git = (*this->InternalSliceGUIMap).find(layoutName);
    git++;

    if ( git != gend)
      return ((*git).first.c_str());
    else
      return NULL;
    }
  else
    return NULL;
}

const char* vtkSlicerSlicesGUI::GetNthSliceGUILayoutName(int n)
{
  if (this->InternalSliceGUIMap && n >= 0 && n < (int)(*this->InternalSliceGUIMap).size())
    {
    SliceGUIMap::const_iterator git;
    int i;
    for (git = (*this->InternalSliceGUIMap).begin(), i=0; i < n; ++i, ++git)
      {
      }

    return ((*git).first.c_str());
    }
  else
    {
    return NULL;
    }
}

vtkSlicerSliceGUI* vtkSlicerSlicesGUI::GetNthSliceGUI(int n)
{
  if (this->InternalSliceGUIMap && n >= 0 && n < (int)(*this->InternalSliceGUIMap).size())
    {
    SliceGUIMap::const_iterator git;
    int i;
    for (git = (*this->InternalSliceGUIMap).begin(), i=0; i < n; ++i, ++git)
      {
      }

    return ((*git).second);
    }
  else
    {
    return NULL;
    }
}


void vtkSlicerSlicesGUI::UpdateGUI()
{
  vtkSlicerNodeSelectorWidget *sliceNodeSelector = vtkSlicerNodeSelectorWidget::SafeDownCast( (*this->InternalParameterWidgetMap)["SliceNodeSelector"].GetPointer() );

  
  // Need to know the slice node
  vtkMRMLSliceNode* n = vtkMRMLSliceNode::SafeDownCast(sliceNodeSelector->GetSelected());
  if (n == NULL) 
    {
    return;
    }

  vtkKWLabelWithLabel *layoutName = vtkKWLabelWithLabel::SafeDownCast((*this->InternalParameterWidgetMap)["LayoutName"]);
  if (layoutName)
    {
    layoutName->GetWidget()->SetText(n->GetLayoutName());
    }

  vtkKWMenuButtonWithLabel *orientation = vtkKWMenuButtonWithLabel::SafeDownCast((*this->InternalParameterWidgetMap)["Orientation"]);
  if (orientation)
    {
    orientation->GetWidget()->SetValue(n->GetOrientationString());
    }

  vtkKWPushButtonWithLabel *sliceVisible = vtkKWPushButtonWithLabel::SafeDownCast((*this->InternalParameterWidgetMap)["SliceVisible"]);
  if (sliceVisible)
    {
    if (n->GetSliceVisible())
      {
      sliceVisible->GetWidget()->SetImageToIcon(this->VisibilityIcons->GetVisibleIcon());
      }
    else
      {
      sliceVisible->GetWidget()->SetImageToIcon(this->VisibilityIcons->GetInvisibleIcon());
      }
    }

  vtkKWCheckButtonWithLabel *widgetVisible = vtkKWCheckButtonWithLabel::SafeDownCast((*this->InternalParameterWidgetMap)["WidgetVisible"]);
  if (widgetVisible)
    {
    widgetVisible->GetWidget()->SetSelectedState(n->GetWidgetVisible());
    }

  vtkKWLabel *dimensionX = vtkKWLabel::SafeDownCast((*this->InternalParameterWidgetMap)["DimensionX"]);
  if (dimensionX)
    {
    std::stringstream ss;
    ss << n->GetDimensions()[0];
    dimensionX->SetText( ss.str().c_str() );
    }

  vtkKWLabel *dimensionY = vtkKWLabel::SafeDownCast((*this->InternalParameterWidgetMap)["DimensionY"]);
  if (dimensionY)
    {
    std::stringstream ss;
    ss << n->GetDimensions()[1];
    dimensionY->SetText( ss.str().c_str() );
    }

  vtkKWLabel *dimensionZ = vtkKWLabel::SafeDownCast((*this->InternalParameterWidgetMap)["DimensionZ"]);
  if (dimensionZ)
    {
    std::stringstream ss;
    ss << n->GetDimensions()[2];
    dimensionZ->SetText( ss.str().c_str() );
    }
    
  vtkKWLabel *fieldOfViewX = vtkKWLabel::SafeDownCast((*this->InternalParameterWidgetMap)["FieldOfViewX"]);
  if (fieldOfViewX)
    {
    std::stringstream ss;
    ss << n->GetFieldOfView()[0];
    fieldOfViewX->SetText( ss.str().c_str() );
    }

  vtkKWLabel *fieldOfViewY = vtkKWLabel::SafeDownCast((*this->InternalParameterWidgetMap)["FieldOfViewY"]);
  if (fieldOfViewY)
    {
    std::stringstream ss;
    ss << n->GetFieldOfView()[1];
    fieldOfViewY->SetText( ss.str().c_str() );
    }

  vtkKWLabel *fieldOfViewZ = vtkKWLabel::SafeDownCast((*this->InternalParameterWidgetMap)["FieldOfViewZ"]);
  if (fieldOfViewZ)
    {
    std::stringstream ss;
    ss << n->GetFieldOfView()[2];
    fieldOfViewZ->SetText( ss.str().c_str() );
    }


  vtkKWEntry *layoutRows = vtkKWEntry::SafeDownCast((*this->InternalParameterWidgetMap)["LayoutRows"]);
  if (layoutRows)
    {
    std::stringstream ss;
    ss << n->GetLayoutGridRows();
    layoutRows->SetValue( ss.str().c_str() );
    }

  vtkKWEntry *layoutColumns = vtkKWEntry::SafeDownCast((*this->InternalParameterWidgetMap)["LayoutColumns"]);
  if (layoutColumns)
    {
    std::stringstream ss;
    ss << n->GetLayoutGridColumns();
    layoutColumns->SetValue( ss.str().c_str() );
    }

  vtkKWRadioButtonSetWithLabel *sliceSpacingMode = vtkKWRadioButtonSetWithLabel::SafeDownCast((*this->InternalParameterWidgetMap)["SliceSpacingMode"]);
  if (sliceSpacingMode)
    {
    if (n->GetSliceSpacingMode() == vtkMRMLSliceNode::AutomaticSliceSpacingMode)
      {
      vtkKWRadioButton* rb = sliceSpacingMode->GetWidget()->GetWidget(0);
      rb->SetSelectedState(1);
      }
    else if (n->GetSliceSpacingMode() == vtkMRMLSliceNode::PrescribedSliceSpacingMode)
      {
      vtkKWRadioButton* rb = sliceSpacingMode->GetWidget()->GetWidget(1);
      rb->SetSelectedState(1);
      }
    }
    
  vtkKWEntryWithLabel *prescribedSpacing = vtkKWEntryWithLabel::SafeDownCast((*this->InternalParameterWidgetMap)["PrescribedSpacing"]);
  if (prescribedSpacing)
    {
    std::stringstream ss;
    ss << n->GetPrescribedSliceSpacing()[2];
    prescribedSpacing->GetWidget()->SetValue( ss.str().c_str() );
    }
  
}


//---------------------------------------------------------------------------
void vtkSlicerSlicesGUI::ProcessGUIEvents ( vtkObject *caller,
                                            unsigned long event, void *callData)
{
  // std::cout << "In Process GUI Events" << std::endl;
  
  vtkSlicerNodeSelectorWidget *selector = vtkSlicerNodeSelectorWidget::SafeDownCast(caller);
  vtkSlicerNodeSelectorWidget *sliceNodeSelector = vtkSlicerNodeSelectorWidget::SafeDownCast( (*this->InternalParameterWidgetMap)["SliceNodeSelector"].GetPointer() );

  vtkKWEntry *entry = vtkKWEntry::SafeDownCast(caller);
  vtkKWPushButton *push = vtkKWPushButton::SafeDownCast(caller);
  vtkKWCheckButton *check = vtkKWCheckButton::SafeDownCast(caller);
  vtkKWRadioButton *radio = vtkKWRadioButton::SafeDownCast(caller);
  vtkKWMenu *menu = vtkKWMenu::SafeDownCast(caller);
  
  
  vtkKWPushButtonWithLabel *sliceVisible = vtkKWPushButtonWithLabel::SafeDownCast((*this->InternalParameterWidgetMap)["SliceVisible"]);
  vtkKWCheckButtonWithLabel *widgetVisible = vtkKWCheckButtonWithLabel::SafeDownCast((*this->InternalParameterWidgetMap)["WidgetVisible"]);
  vtkKWEntry *layoutRows = vtkKWEntry::SafeDownCast((*this->InternalParameterWidgetMap)["LayoutRows"]);
  vtkKWEntry *layoutColumns = vtkKWEntry::SafeDownCast((*this->InternalParameterWidgetMap)["LayoutColumns"]);
  vtkKWEntryWithLabel *prescribedSpacing = vtkKWEntryWithLabel::SafeDownCast((*this->InternalParameterWidgetMap)["PrescribedSpacing"]);
  vtkKWMenuButtonWithLabel *orientation = vtkKWMenuButtonWithLabel::SafeDownCast((*this->InternalParameterWidgetMap)["Orientation"]);

  // Need to know the slice node
  vtkMRMLSliceNode* n = vtkMRMLSliceNode::SafeDownCast(sliceNodeSelector->GetSelected());
  if (n == NULL) 
    {
    return;
    }

  
  if (selector && selector == sliceNodeSelector
      && event == vtkSlicerNodeSelectorWidget::NodeSelectedEvent ) 
    {
    // Add/remove observers on the slice node to this GUI
    if (this->SliceNode)
      {
      this->SliceNode->RemoveObservers( vtkCommand::ModifiedEvent, (vtkCommand *)this->MRMLCallbackCommand);
      }
    n->AddObserver(vtkCommand::ModifiedEvent, (vtkCommand *)this->MRMLCallbackCommand);
    this->SliceNode = n;

    // Selected a new slice node.  Fill in the GUI with the
    // information from the slice node
    this->UpdateGUI();

    }
  else if (menu && orientation
           && menu == orientation->GetWidget()->GetMenu()
           && event == vtkKWMenu::MenuItemInvokedEvent)
    {
    if (!strcmp(orientation->GetWidget()->GetValue(), "Axial"))
      {
      n->SetOrientationToAxial();
      }
    else if (!strcmp(orientation->GetWidget()->GetValue(), "Sagittal"))
      {
      n->SetOrientationToSagittal();
      }
    else if (!strcmp(orientation->GetWidget()->GetValue(), "Coronal"))
      {
      n->SetOrientationToCoronal();
      }
    else if (!strcmp(orientation->GetWidget()->GetValue(), "Reformat"))
      {
      n->SetOrientationToReformat();
      }
    }
  else if (push && sliceVisible
           && push == sliceVisible->GetWidget()
           && event == vtkKWPushButton::InvokedEvent )
    {
    n->SetSliceVisible( !n->GetSliceVisible() );
    }
  else if (check && widgetVisible
           && check == widgetVisible->GetWidget()
           && event == vtkKWCheckButton::SelectedStateChangedEvent )
    {
    n->SetWidgetVisible( check->GetSelectedState() );
    }
  else if (entry && entry == layoutRows
           && event == vtkKWEntry::EntryValueChangedEvent)
    {
    std::stringstream ss;
    ss << entry->GetValue();
    int val;
    ss >> val;
    n->SetLayoutGridRows(val);
    }
  else if (entry && entry == layoutColumns
           && event == vtkKWEntry::EntryValueChangedEvent)
    {
    std::stringstream ss;
    ss << entry->GetValue();
    int val;
    ss >> val;
    n->SetLayoutGridColumns(val);
    }
  else if (entry && prescribedSpacing
           && entry == prescribedSpacing->GetWidget()
           && event == vtkKWEntry::EntryValueChangedEvent)
    {
    std::stringstream ss;
    ss << entry->GetValue();
    double val;
    ss >> val;
    double *current = n->GetPrescribedSliceSpacing();
    double spacing[3];
    spacing[0] = current[0];
    spacing[1] = current[1];
    spacing[2] = val;
    n->SetPrescribedSliceSpacing(spacing);
    }
  else if (radio)
    {
    if (!strcmp(radio->GetValue(), "Automatic"))
      {
      n->SetSliceSpacingModeToAutomatic();
      }
    else if (!strcmp(radio->GetValue(), "Prescribed"))
      {
      n->SetSliceSpacingModeToPrescribed();
      }
    }


}


//---------------------------------------------------------------------------
void vtkSlicerSlicesGUI::ProcessLogicEvents ( vtkObject *caller,
                                              unsigned long event, void *callData )
{
  // Fill in
}
 

//---------------------------------------------------------------------------
void vtkSlicerSlicesGUI::ProcessMRMLEvents ( vtkObject *caller,
                                             unsigned long event, void *callData )
{
  // std::cout << "In ProcessMRMLEvents" << std::endl;
  this->UpdateGUI();
}
 

//---------------------------------------------------------------------------
void vtkSlicerSlicesGUI::Enter ( )
{
  // Fill in
}

//---------------------------------------------------------------------------
void vtkSlicerSlicesGUI::Exit ( )
{
  // Fill in
}


//---------------------------------------------------------------------------
void vtkSlicerSlicesGUI::BuildGUI (  )
{
  vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
    
  // ---
  // MODULE GUI FRAME 
  // configure a page for a volume loading UI for now.
  // later, switch on the modulesButton in the SlicerControlGUI
  // ---
  // create a page
  this->UIPanel->AddPage ( "Slices", "Slices", NULL );
    
  // Define your help text and build the help frame here.
  const char *help = "The Slices Module manages the display of the Slice Viewers.";
  const char *about = "This work was supported by NA-MIC, NAC, BIRN, NCIGT, and the Slicer Community. See <a>http://www.slicer.org</a> for details. ";
  vtkKWWidget *page = this->UIPanel->GetPageWidget ( "Slices" );
  this->BuildHelpAndAboutFrame ( page, help, about );

  // ---
  // DISPLAY FRAME            
  vtkSlicerModuleCollapsibleFrame *sliceDisplayFrame = vtkSlicerModuleCollapsibleFrame::New ( );
  sliceDisplayFrame->SetParent ( this->UIPanel->GetPageWidget ( "Slices" ) );
  sliceDisplayFrame->Create ( );
  sliceDisplayFrame->SetLabelText ("Slice information");
  //sliceDisplayFrame->CollapseFrame ( );
  app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                sliceDisplayFrame->GetWidgetName(), this->UIPanel->GetPageWidget("Slices")->GetWidgetName());

  sliceDisplayFrame->Delete();

  // Active slice selector
  vtkSlicerNodeSelectorWidget *sliceNodeSelector =
    vtkSlicerNodeSelectorWidget::New();
  sliceNodeSelector->SetParent( sliceDisplayFrame->GetFrame() );
  sliceNodeSelector->Create();
  sliceNodeSelector->ShowHiddenOn();
  sliceNodeSelector->SetNodeClass("vtkMRMLSliceNode", NULL, NULL, NULL);
  sliceNodeSelector->SetMRMLScene(this->GetMRMLScene());
  sliceNodeSelector->SetBorderWidth(2);
  sliceNodeSelector->GetWidget()->GetWidget()->IndicatorVisibilityOff();
  sliceNodeSelector->GetWidget()->GetWidget()->SetWidth(24);
  sliceNodeSelector->SetLabelText( "Active Slice: ");
  sliceNodeSelector->SetBalloonHelpString("Select a slice from the current scene.");
  this->Script("pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
               sliceNodeSelector->GetWidgetName());

  (*this->InternalParameterWidgetMap)["SliceNodeSelector"]
    = sliceNodeSelector;
  sliceNodeSelector->Delete();

  // Layout name
  vtkKWLabelWithLabel *layoutName = vtkKWLabelWithLabel::New();
  layoutName->SetParent( sliceDisplayFrame->GetFrame() );
  layoutName->Create();
  layoutName->SetLabelText("Layout name: ");
  layoutName->GetWidget()->SetText("(none)");
  layoutName->SetBalloonHelpString("Name of the slice.");
  this->Script("pack %s -side top -anchor ne -padx 2 -pady 2",
               layoutName->GetWidgetName());
  (*this->InternalParameterWidgetMap)["LayoutName"]
    = layoutName;
  layoutName->Delete();

  // Orientation
  vtkKWMenuButtonWithLabel *orientation = vtkKWMenuButtonWithLabel::New();
  orientation->SetParent( sliceDisplayFrame->GetFrame() );
  orientation->Create();
  orientation->SetLabelText("Orientation: ");
  orientation->SetBalloonHelpString("Slice orientation (axial, sagittal, coronal, reformat).");
  orientation->GetWidget()->GetMenu()->AddRadioButton("Axial");
  orientation->GetWidget()->GetMenu()->AddRadioButton("Sagittal");
  orientation->GetWidget()->GetMenu()->AddRadioButton("Coronal");
  orientation->GetWidget()->GetMenu()->AddRadioButton("Reformat");
  orientation->GetWidget()->SetValue("Axial");
  this->Script("pack %s -side top -anchor ne -padx 2 -pady 2",
               orientation->GetWidgetName());
  (*this->InternalParameterWidgetMap)["Orientation"]
    = orientation;
  orientation->Delete();


  // Make sure we have the visibility icons available
  if (!this->VisibilityIcons)
    {
    this->VisibilityIcons = vtkSlicerVisibilityIcons::New();
    }
    
    
  // Slice visible
  vtkKWPushButtonWithLabel *sliceVisible = vtkKWPushButtonWithLabel::New();
  sliceVisible->SetParent( sliceDisplayFrame->GetFrame() );
  sliceVisible->Create();
  sliceVisible->SetLabelText("Slice visible: ");
  sliceVisible->GetWidget()->SetReliefToFlat();
  sliceVisible->GetWidget()->SetBorderWidth( 0 );
  sliceVisible->GetWidget()->SetImageToIcon( this->VisibilityIcons->GetInvisibleIcon() );
  sliceVisible->SetBalloonHelpString("Toggle the visibility of the slice in the 3D scene");
  this->Script("pack %s -side top -anchor ne -padx 2 -pady 2",
               sliceVisible->GetWidgetName());
  (*this->InternalParameterWidgetMap)["SliceVisible"]
    = sliceVisible;
  sliceVisible->Delete();

  // Widget visible
  vtkKWCheckButtonWithLabel *widgetVisible = vtkKWCheckButtonWithLabel::New();
  widgetVisible->SetParent( sliceDisplayFrame->GetFrame() );
  widgetVisible->Create();
  widgetVisible->SetLabelText("Widget visible: ");
  widgetVisible->GetWidget()->SetSelectedState( 0 );
  widgetVisible->SetBalloonHelpString("Toggle the visibility of the reformat widget in the 3D scene");
  this->Script("pack %s -side top -anchor ne -padx 2 -pady 2",
               widgetVisible->GetWidgetName());
  (*this->InternalParameterWidgetMap)["WidgetVisible"]
    = widgetVisible;
  widgetVisible->Delete();

  // Dimension
  vtkKWFrame *dimensionFrame = vtkKWFrame::New();
  dimensionFrame->SetParent( sliceDisplayFrame->GetFrame() );
  dimensionFrame->Create();
  this->Script("pack %s -side top -anchor ne -padx 2 -pady 2",
               dimensionFrame->GetWidgetName());
  (*this->InternalParameterWidgetMap)["FieldOfViewFrame"]
    = dimensionFrame;
  dimensionFrame->Delete();
                 
  vtkKWLabel *dimensionLabel = vtkKWLabel::New();
  dimensionLabel->SetParent( dimensionFrame );
  dimensionLabel->Create();
  dimensionLabel->SetText( "Dimension: " );
  dimensionLabel->SetBalloonHelpString("Dimension of the slice.");
  (*this->InternalParameterWidgetMap)["DimensionLabel"]
    = dimensionLabel;
  dimensionLabel->Delete();

  vtkKWLabel *dimensionX = vtkKWLabel::New();
  dimensionX->SetParent( dimensionFrame );
  dimensionX->Create();
  dimensionX->SetText("100");
  dimensionX->SetBalloonHelpString("Slice dimension in X.");
  (*this->InternalParameterWidgetMap)["DimensionX"]
    = dimensionX;
  dimensionX->Delete();

  vtkKWLabel *dimensionY = vtkKWLabel::New();
  dimensionY->SetParent( dimensionFrame );
  dimensionY->Create();
  dimensionY->SetText("100");
  dimensionY->SetBalloonHelpString("Slice dimension in Y.");
  (*this->InternalParameterWidgetMap)["DimensionY"]
    = dimensionY;
  dimensionY->Delete();

  vtkKWLabel *dimensionZ = vtkKWLabel::New();
  dimensionZ->SetParent( dimensionFrame );
  dimensionZ->Create();
  dimensionZ->SetText("1");
  dimensionZ->SetBalloonHelpString("Slice dimension in Z.");
  (*this->InternalParameterWidgetMap)["DimensionZ"]
    = dimensionZ;
  dimensionZ->Delete();
    
  this->Script("pack %s %s %s %s -side left -anchor ne -padx 2 -pady 2",
               dimensionLabel->GetWidgetName(), dimensionX->GetWidgetName(),
               dimensionY->GetWidgetName(), dimensionZ->GetWidgetName() );

  // FieldOfView 
  vtkKWFrame *fieldOfViewFrame = vtkKWFrame::New();
  fieldOfViewFrame->SetParent( sliceDisplayFrame->GetFrame() );
  fieldOfViewFrame->Create();
  this->Script("pack %s -side top -anchor ne -padx 2 -pady 2",
               fieldOfViewFrame->GetWidgetName());
  (*this->InternalParameterWidgetMap)["FieldOfViewFrame"]
    = fieldOfViewFrame;
  fieldOfViewFrame->Delete();
                 
  vtkKWLabel *fieldOfViewLabel = vtkKWLabel::New();
  fieldOfViewLabel->SetParent( fieldOfViewFrame );
  fieldOfViewLabel->Create();
  fieldOfViewLabel->SetText( "Field of view: " );
  fieldOfViewLabel->SetBalloonHelpString("Field of view of the slice.");
  (*this->InternalParameterWidgetMap)["FieldOfViewLabel"]
    = fieldOfViewLabel;
  fieldOfViewLabel->Delete();

  vtkKWLabel *fieldOfViewX = vtkKWLabel::New();
  fieldOfViewX->SetParent( fieldOfViewFrame );
  fieldOfViewX->Create();
  fieldOfViewX->SetText("50");
  fieldOfViewX->SetBalloonHelpString("Field of view in X (mm).");
  (*this->InternalParameterWidgetMap)["FieldOfViewX"]
    = fieldOfViewX;
  fieldOfViewX->Delete();

  vtkKWLabel *fieldOfViewY = vtkKWLabel::New();
  fieldOfViewY->SetParent( fieldOfViewFrame );
  fieldOfViewY->Create();
  fieldOfViewY->SetText("50");
  fieldOfViewY->SetBalloonHelpString("Field of view in Y (mm).");
  (*this->InternalParameterWidgetMap)["FieldOfViewY"]
    = fieldOfViewY;
  fieldOfViewY->Delete();

  vtkKWLabel *fieldOfViewZ = vtkKWLabel::New();
  fieldOfViewZ->SetParent( fieldOfViewFrame );
  fieldOfViewZ->Create();
  fieldOfViewZ->SetText("1");
  fieldOfViewZ->SetBalloonHelpString("Field of view in Z (mm).");
  (*this->InternalParameterWidgetMap)["FieldOfViewZ"]
    = fieldOfViewZ;
  fieldOfViewZ->Delete();
    
  this->Script("pack %s %s %s %s -side left -anchor ne -padx 2 -pady 2",
               fieldOfViewLabel->GetWidgetName(), fieldOfViewX->GetWidgetName(),
               fieldOfViewY->GetWidgetName(), fieldOfViewZ->GetWidgetName() );

  // Layout grid
  vtkKWFrame *layoutFrame = vtkKWFrame::New();
  layoutFrame->SetParent( sliceDisplayFrame->GetFrame() );
  layoutFrame->Create();
  this->Script("pack %s -side top -anchor ne -padx 2 -pady 2",
               layoutFrame->GetWidgetName());
  (*this->InternalParameterWidgetMap)["LayoutFrame"]
    = layoutFrame;
  layoutFrame->Delete();
                 
  vtkKWLabel *layoutLabel = vtkKWLabel::New();
  layoutLabel->SetParent( layoutFrame );
  layoutLabel->Create();
  layoutLabel->SetText( "Lightbox layout: " );
  layoutLabel->SetBalloonHelpString("Layout of the lightbox (rows, columns)");
  (*this->InternalParameterWidgetMap)["LayoutLabel"]
    = layoutLabel;
  layoutLabel->Delete();

  vtkKWEntry *layoutRows = vtkKWEntry::New();
  layoutRows->SetParent( layoutFrame );
  layoutRows->Create();
  layoutRows->SetWidth(4);
  layoutRows->SetValue("1");
  layoutRows->SetBalloonHelpString("Number of rows in the lightbox.");
  (*this->InternalParameterWidgetMap)["LayoutRows"]
    = layoutRows;
  layoutRows->Delete();
    
  vtkKWEntry *layoutColumns = vtkKWEntry::New();
  layoutColumns->SetParent( layoutFrame );
  layoutColumns->Create();
  layoutColumns->SetWidth(4);
  layoutColumns->SetValue("1");
  layoutColumns->SetBalloonHelpString("Number of columns in the lightbox.");
  (*this->InternalParameterWidgetMap)["LayoutColumns"]
    = layoutColumns;
  layoutColumns->Delete();

  this->Script("pack %s %s %s -side left -anchor ne -padx 2 -pady 2",
               layoutLabel->GetWidgetName(), layoutRows->GetWidgetName(),
               layoutColumns->GetWidgetName() );
    
  // Slice spacing mode
  vtkKWRadioButtonSetWithLabel *sliceSpacingMode = vtkKWRadioButtonSetWithLabel::New();
  sliceSpacingMode->SetParent( sliceDisplayFrame->GetFrame() );
  sliceSpacingMode->Create();
  sliceSpacingMode->SetLabelText("Slice spacing mode: ");
  sliceSpacingMode->GetWidget()->PackHorizontallyOn();
  sliceSpacingMode->SetBalloonHelpString("Slice spacing can be prescribed by the user or context or set automatically.");

  vtkKWRadioButton *automatic = sliceSpacingMode->GetWidget()->AddWidget(0);
  automatic->SetValue("Automatic");
  automatic->SetText("Automatic");
  automatic->SetAnchorToWest();
  automatic->SetSelectedState(1);

  vtkKWRadioButton *prescribed = sliceSpacingMode->GetWidget()->AddWidget(1);
  prescribed->SetValue("Prescribed");
  prescribed->SetText("Prescribed");
  prescribed->SetAnchorToWest();
  prescribed->SetSelectedState(0);

  this->Script("pack %s -side top -anchor ne -padx 2 -pady 2",
               sliceSpacingMode->GetWidgetName());
  (*this->InternalParameterWidgetMap)["SliceSpacingMode"]
    = sliceSpacingMode;
  sliceSpacingMode->Delete();
    

  // Prescribed slice spacing
  vtkKWEntryWithLabel *prescribedSpacing = vtkKWEntryWithLabel::New();
  prescribedSpacing->SetParent( sliceDisplayFrame->GetFrame() );
  prescribedSpacing->Create();
  prescribedSpacing->GetWidget()->SetWidth(7);
  prescribedSpacing->SetLabelText("Prescribed spacing: ");
  prescribedSpacing->GetWidget()->SetValue("1.0");
  prescribedSpacing->SetBalloonHelpString("Slice spacing used when slice spacing mode is prescribed by the user or by the context.");
  this->Script("pack %s -side top -anchor ne -padx 2 -pady 2",
               prescribedSpacing->GetWidgetName());
  (*this->InternalParameterWidgetMap)["PrescribedSpacing"]
    = prescribedSpacing;
  prescribedSpacing->Delete();
    
}


//----------------------------------------------------------------------------
void vtkSlicerSlicesGUI::PrintSelf(ostream& os, vtkIndent indent)
{
  this->vtkObject::PrintSelf(os, indent);
  os << indent << "SlicerSlicesGUI:" << this->GetClassName ( ) << "\n";
  os << indent << "SliceGUIMap: " << this->InternalSliceGUIMap << endl;
  os << indent << "SliceParameterMap: " << this->InternalParameterWidgetMap << endl;
}
