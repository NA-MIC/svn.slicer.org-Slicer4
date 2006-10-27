#include "vtkObject.h"
#include "vtkObjectFactory.h"
#include "vtkCommand.h"

#include "vtkSlicerVolumesGUI.h"
#include "vtkSlicerVolumesLogic.h"
#include "vtkSlicerApplication.h"
#include "vtkMRMLVolumeNode.h"
#include "vtkSlicerModuleCollapsibleFrame.h"

#include "vtkKWWidget.h"
#include "vtkKWMenuButton.h"
#include "vtkKWCheckButton.h"
#include "vtkKWPushButton.h"
#include "vtkKWMenu.h"
#include "vtkKWFrameWithLabel.h"
#include "vtkKWFrame.h"
#include "vtkKWLoadSaveButton.h"
#include "vtkKWLoadSaveButtonWithLabel.h"
#include "vtkKWLoadSaveDialog.h"
#include "vtkKWEntry.h"
#include "vtkKWEntryWithLabel.h"

//---------------------------------------------------------------------------
vtkStandardNewMacro (vtkSlicerVolumesGUI );
vtkCxxRevisionMacro ( vtkSlicerVolumesGUI, "$Revision: 1.0 $");


//---------------------------------------------------------------------------
vtkSlicerVolumesGUI::vtkSlicerVolumesGUI ( )
{

    this->Logic = NULL;
    this->VolumeNode = NULL;
    this->SelectedVolumeID = NULL;

    this->VolumeSelectorWidget = NULL;
    this->LoadVolumeButton = NULL;
    this->SaveVolumeButton = NULL;
    this->VolumeDisplayWidget = NULL;

    this->NameEntry = NULL;
    this->CenterImageMenu = NULL;
    this->LabelMapCheckButton = NULL;
    this->ApplyButton=NULL;
}


//---------------------------------------------------------------------------
vtkSlicerVolumesGUI::~vtkSlicerVolumesGUI ( )
{
  this->RemoveGUIObservers();
    
  if (this->SelectedVolumeID)
    {
    delete [] this->SelectedVolumeID;
    this->SelectedVolumeID = NULL;
    }
  if (this->LoadVolumeButton )
    {
    this->LoadVolumeButton->SetParent(NULL );
    this->LoadVolumeButton->Delete ( );
    }
  if (this->SaveVolumeButton )
    {
    this->SaveVolumeButton->SetParent(NULL );
    this->SaveVolumeButton->Delete ( );
    }

  if (this->VolumeDisplayWidget)
    {
    this->VolumeDisplayWidget->SetParent(NULL );
    this->VolumeDisplayWidget->Delete ( );
    }
  if (this->VolumeSelectorWidget)
    {
    this->VolumeSelectorWidget->SetParent(NULL );
    this->VolumeSelectorWidget->Delete ( );
    }
  if (this->CenterImageMenu)
    {
    this->CenterImageMenu->SetParent(NULL );
    this->CenterImageMenu->Delete ( );
    }
  if (this->LabelMapCheckButton)
    {
    this->LabelMapCheckButton->SetParent(NULL );
    this->LabelMapCheckButton->Delete ( );
    }
  if (this->ApplyButton)
    {
    this->ApplyButton->SetParent(NULL );
    this->ApplyButton->Delete ( );
    }
  if (this->NameEntry)
    {
    this->NameEntry->SetParent(NULL );
    this->NameEntry->Delete ( );
    }

  this->SetModuleLogic ( NULL );
   vtkSetMRMLNodeMacro (this->VolumeNode, NULL );
}


//---------------------------------------------------------------------------
void vtkSlicerVolumesGUI::PrintSelf ( ostream& os, vtkIndent indent )
{
    this->vtkObject::PrintSelf ( os, indent );

    os << indent << "SlicerVolumesGUI: " << this->GetClassName ( ) << "\n";
    os << indent << "VolumeNode: " << this->GetVolumeNode ( ) << "\n";
    os << indent << "Logic: " << this->GetLogic ( ) << "\n";
    // print widgets?
}


//---------------------------------------------------------------------------
void vtkSlicerVolumesGUI::RemoveGUIObservers ( )
{
    // Fill in
    if (this->LoadVolumeButton)
      {
      this->LoadVolumeButton->GetWidget()->RemoveObservers ( vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );
      }
    if (this->SaveVolumeButton)
      {
      this->SaveVolumeButton->RemoveObservers ( vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );
      }
    if (this->ApplyButton)
      {
      this->ApplyButton->RemoveObservers (vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
      }
}


//---------------------------------------------------------------------------
void vtkSlicerVolumesGUI::AddGUIObservers ( )
{

    // Fill in
    // observer load volume button
    this->LoadVolumeButton->GetWidget()->AddObserver ( vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
    this->SaveVolumeButton->AddObserver ( vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
    this->ApplyButton->AddObserver (vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
}



//---------------------------------------------------------------------------
void vtkSlicerVolumesGUI::ProcessGUIEvents ( vtkObject *caller,
                                             unsigned long event, void *callData )
{
  if (this->LoadVolumeButton->GetWidget() == vtkKWLoadSaveButton::SafeDownCast(caller) && event == vtkKWPushButton::InvokedEvent )
    {
    const char * filename = this->LoadVolumeButton->GetWidget()->GetFileName();
    if (filename)
      {
      const vtksys_stl::string fname(filename);
      vtksys_stl::string name = vtksys::SystemTools::GetFilenameName(fname);
      this->NameEntry->GetWidget()->SetValue(name.c_str());
      }
    else
      {
      this->NameEntry->GetWidget()->SetValue("");
      }
    this->LoadVolumeButton->GetWidget()->SetText ("Select Volume File");
    }  
  else if (this->ApplyButton == vtkKWPushButton::SafeDownCast(caller)  && event == vtkKWPushButton::InvokedEvent )
    {
    // If a file has been selected for loading...
    char *fileName = this->LoadVolumeButton->GetWidget()->GetFileName();
    if ( fileName ) 
      {

       vtkKWMenuButton *mb = this->CenterImageMenu->GetWidget();
       int centered;
       if ( !strcmp (mb->GetValue(), "Centered") )   
         {
         centered = 1;
         }
       else 
         {
         centered = 0;
         }

       int labelMap;
       if ( this->LabelMapCheckButton->GetSelectedState() )
         {
         labelMap = 1;
         }
       else
         {
         labelMap = 0;
         }

      vtkSlicerVolumesLogic* volumeLogic = this->Logic;
      vtkMRMLVolumeNode *volumeNode = volumeLogic->AddArchetypeVolume( fileName, centered, labelMap, this->NameEntry->GetWidget()->GetValue() );
      if ( volumeNode == NULL ) 
        {
        //TODO: generate an error...
        }
      else
        {
        this->LoadVolumeButton->GetWidget()->GetLoadSaveDialog()->SaveLastPathToRegistry("OpenPath");

        this->ApplicationLogic->GetSelectionNode()->SetActiveVolumeID( volumeNode->GetID() );
        this->ApplicationLogic->PropagateVolumeSelection();
        this->VolumeDisplayWidget->SetVolumeNode(volumeNode);               
        }
      }
      return;
    }
    else if (this->SaveVolumeButton == vtkKWLoadSaveButton::SafeDownCast(caller) && event == vtkKWPushButton::InvokedEvent )
      {
      // If a file has been selected for saving...
      char *fileName = this->SaveVolumeButton->GetFileName();
      if ( fileName ) 
      {
        vtkSlicerVolumesLogic* volumeLogic = this->Logic;
        vtkMRMLVolumeNode *volNode = vtkMRMLVolumeNode::SafeDownCast(this->VolumeSelectorWidget->GetSelected());
        if ( !volumeLogic->SaveArchetypeVolume( fileName, volNode ))
          {
         // TODO: generate an error...
          }
        else
          {
          this->SaveVolumeButton->GetLoadSaveDialog()->SaveLastPathToRegistry("OpenPath");           
          }
       }
       return;
    }
} 



//---------------------------------------------------------------------------
void vtkSlicerVolumesGUI::ProcessLogicEvents ( vtkObject *caller,
                                               unsigned long event, void *callData )
{
    // Fill in
}

//---------------------------------------------------------------------------
void vtkSlicerVolumesGUI::ProcessMRMLEvents ( vtkObject *caller,
                                              unsigned long event, void *callData )
{
    // Fill in
}


//---------------------------------------------------------------------------
void vtkSlicerVolumesGUI::Enter ( )
{
    // Fill in
}

//---------------------------------------------------------------------------
void vtkSlicerVolumesGUI::Exit ( )
{
    // Fill in
}


//---------------------------------------------------------------------------
void vtkSlicerVolumesGUI::BuildGUI ( )
{

  vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
  // Define your help text here.
  const char *help = "**Volumes Module:** Load, save and adjust display parameters of volume data. ";

    // ---
    // MODULE GUI FRAME 
    // configure a page for a volume loading UI for now.
    // later, switch on the modulesButton in the SlicerControlGUI
    // ---
    // create a page
    this->UIPanel->AddPage ( "Volumes", "Volumes", NULL );
    
    vtkKWWidget *page = this->UIPanel->GetPageWidget ( "Volumes" );

    // HELP FRAME
    vtkSlicerModuleCollapsibleFrame *volHelpFrame = vtkSlicerModuleCollapsibleFrame::New ( );
    volHelpFrame->SetParent ( page );
    volHelpFrame->Create ( );
    volHelpFrame->CollapseFrame ( );
    volHelpFrame->SetLabelText ("Help");
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                  volHelpFrame->GetWidgetName(), page->GetWidgetName());

    // configure the parent classes help text widget
    this->HelpText->SetParent ( volHelpFrame->GetFrame() );
    this->HelpText->Create ( );
    this->HelpText->SetHorizontalScrollbarVisibility ( 0 );
    this->HelpText->SetVerticalScrollbarVisibility ( 1 );
    this->HelpText->GetWidget()->SetText ( help );
    this->HelpText->GetWidget()->SetReliefToFlat ( );
    this->HelpText->GetWidget()->SetWrapToWord ( );
    this->HelpText->GetWidget()->ReadOnlyOn ( );
    this->HelpText->GetWidget()->QuickFormattingOn ( );
    this->HelpText->GetWidget()->SetBalloonHelpString ( "" );
    app->Script ( "pack %s -side top -fill x -expand y -anchor w -padx 2 -pady 4",
                  this->HelpText->GetWidgetName ( ) );

    // ---
    // LOAD FRAME            
    vtkSlicerModuleCollapsibleFrame *volLoadFrame = vtkSlicerModuleCollapsibleFrame::New ( );
    volLoadFrame->SetParent ( page );
    volLoadFrame->Create ( );
    volLoadFrame->SetLabelText ("Load");
    volLoadFrame->ExpandFrame ( );
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                  volLoadFrame->GetWidgetName(), page->GetWidgetName());
    // add a file browser 
    this->LoadVolumeButton = vtkKWLoadSaveButtonWithLabel::New ( );
    this->LoadVolumeButton->SetParent ( volLoadFrame->GetFrame() );
    this->LoadVolumeButton->Create ( );
    this->LoadVolumeButton->SetWidth(20);
    this->LoadVolumeButton->GetWidget()->SetText ("Select Volume File");
    this->LoadVolumeButton->GetWidget()->GetLoadSaveDialog()->SetFileTypes(
                                                              "{ {volume} {*.*} }");
    this->LoadVolumeButton->GetWidget()->GetLoadSaveDialog()->RetrieveLastPathFromRegistry(
      "OpenPath");
    app->Script("pack %s -side top -anchor nw -expand n -padx 2 -pady 2", 
                this->LoadVolumeButton->GetWidgetName());
    // volume name
    this->NameEntry = vtkKWEntryWithLabel::New();
    this->NameEntry->SetParent(volLoadFrame->GetFrame());
    this->NameEntry->Create();
    this->NameEntry->SetWidth(20);
    this->NameEntry->SetLabelWidth(12);
    this->NameEntry->SetLabelText("Volume Name:");
    this->NameEntry->GetWidget()->SetValue ( "" );
    this->Script(
      "pack %s -side top -anchor nw -expand n -padx 2 -pady 2", 
      this->NameEntry->GetWidgetName());

    // center image
    this->CenterImageMenu = vtkKWMenuButtonWithLabel::New();
    this->CenterImageMenu->SetParent(volLoadFrame->GetFrame());
    this->CenterImageMenu->Create();
    this->CenterImageMenu->SetWidth(20);
    this->CenterImageMenu->SetLabelWidth(12);
    this->CenterImageMenu->SetLabelText("Image Origin:");
    this->CenterImageMenu->GetWidget()->GetMenu()->AddRadioButton ( "Centered");
    this->CenterImageMenu->GetWidget()->GetMenu()->AddRadioButton ( "From File");
    this->CenterImageMenu->GetWidget()->SetValue ( "Centered" );
    this->Script(
      "pack %s -side top -anchor nw -expand n -padx 2 -pady 2", 
      this->CenterImageMenu->GetWidgetName());

    // is this a lable map?
    this->LabelMapCheckButton = vtkKWCheckButton::New();
    this->LabelMapCheckButton->SetParent(volLoadFrame->GetFrame());
    this->LabelMapCheckButton->Create();
    this->LabelMapCheckButton->SelectedStateOff();
    this->LabelMapCheckButton->SetText("Label Map");
    this->Script(
      "pack %s -side top -anchor nw -expand n -padx 2 -pady 2", 
      this->LabelMapCheckButton->GetWidgetName());

    // Apply button
    this->ApplyButton = vtkKWPushButton::New();
    this->ApplyButton->SetParent(volLoadFrame->GetFrame());
    this->ApplyButton->Create();
    this->ApplyButton->SetText("Apply");
    this->ApplyButton->SetWidth ( 20);
    this->Script(
      "pack %s -side top -anchor nw -expand n -padx 2 -pady 2", 
      this->ApplyButton->GetWidgetName());


    // ---
    // DISPLAY FRAME            
    vtkSlicerModuleCollapsibleFrame *volDisplayFrame = vtkSlicerModuleCollapsibleFrame::New ( );    
    volDisplayFrame->SetParent ( page );
    volDisplayFrame->Create ( );
    volDisplayFrame->SetLabelText ("Display");
    volDisplayFrame->CollapseFrame ( );
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                  volDisplayFrame->GetWidgetName(), page->GetWidgetName());

    this->VolumeDisplayWidget = vtkSlicerVolumeDisplayWidget::New ( );
    this->VolumeDisplayWidget->SetMRMLScene(this->GetMRMLScene() );
    this->VolumeDisplayWidget->SetParent ( volDisplayFrame->GetFrame() );
    this->VolumeDisplayWidget->Create ( );
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                  this->VolumeDisplayWidget->GetWidgetName(), volDisplayFrame->GetFrame()->GetWidgetName());

    // ---
    // Save FRAME            
    vtkSlicerModuleCollapsibleFrame *volSaveFrame = vtkSlicerModuleCollapsibleFrame::New ( );
    volSaveFrame->SetParent ( page );
    volSaveFrame->Create ( );
    volSaveFrame->SetLabelText ("Save");
    volSaveFrame->CollapseFrame ( );
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                  volSaveFrame->GetWidgetName(), page->GetWidgetName());

    // selector for save
    this->VolumeSelectorWidget = vtkSlicerNodeSelectorWidget::New() ;
    this->VolumeSelectorWidget->SetParent ( volSaveFrame->GetFrame() );
    this->VolumeSelectorWidget->Create ( );
    this->VolumeSelectorWidget->SetNodeClass("vtkMRMLVolumeNode", NULL, NULL, NULL);
    this->VolumeSelectorWidget->SetMRMLScene(this->GetMRMLScene());
    this->VolumeSelectorWidget->SetBorderWidth(2);
    this->VolumeSelectorWidget->SetPadX(2);
    this->VolumeSelectorWidget->SetPadY(2);
    this->VolumeSelectorWidget->GetWidget()->GetWidget()->IndicatorVisibilityOff();
    this->VolumeSelectorWidget->GetWidget()->GetWidget()->SetWidth(24);
    this->VolumeSelectorWidget->SetLabelText( "Volume To Save: ");
    this->VolumeSelectorWidget->SetBalloonHelpString("select a volume from the current  scene.");
    this->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
                  this->VolumeSelectorWidget->GetWidgetName());

    this->SaveVolumeButton = vtkKWLoadSaveButton::New ( );
    this->SaveVolumeButton->SetParent ( volSaveFrame->GetFrame() );
    this->SaveVolumeButton->Create ( );
    this->SaveVolumeButton->SetText ("Save Volume");
    this->SaveVolumeButton->GetLoadSaveDialog()->SaveDialogOn();
    this->SaveVolumeButton->GetLoadSaveDialog()->SetFileTypes(
                                                              "{ {volume} {*.*} }");
    this->SaveVolumeButton->GetLoadSaveDialog()->RetrieveLastPathFromRegistry(
      "OpenPath");
     app->Script("pack %s -side top -anchor w -padx 2 -pady 4", 
                this->SaveVolumeButton->GetWidgetName());
    
    volLoadFrame->Delete();
    volSaveFrame->Delete();
    volDisplayFrame->Delete ();
    volHelpFrame->Delete();
}





