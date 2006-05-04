#include "vtkObject.h"
#include "vtkObjectFactory.h"
#include "vtkCommand.h"

#include "vtkSlicerVolumesGUI.h"
#include "vtkSlicerVolumesLogic.h"
#include "vtkSlicerApplication.h"
#include "vtkMRMLVolumeNode.h"

#include "vtkKWWidget.h"
#include "vtkKWMenuButton.h"
#include "vtkKWFrameWithLabel.h"
#include "vtkKWFrame.h"

//---------------------------------------------------------------------------
vtkStandardNewMacro (vtkSlicerVolumesGUI );
vtkCxxRevisionMacro ( vtkSlicerVolumesGUI, "$Revision: 1.0 $");


//---------------------------------------------------------------------------
vtkSlicerVolumesGUI::vtkSlicerVolumesGUI ( )
{

    this->Logic = NULL;
    this->VolumeNode = NULL;
    this->SelectedVolumeID = NULL;

    this->LoadVolumeButton = NULL;
    this->VolumeSelectorWidget = NULL;
    this->WindowLevelThresholdEditor = NULL;
}


//---------------------------------------------------------------------------
vtkSlicerVolumesGUI::~vtkSlicerVolumesGUI ( )
{
  if (this->SelectedVolumeID)
    {
    delete [] this->SelectedVolumeID;
    this->SelectedVolumeID = NULL;
    }
    if (this->LoadVolumeButton )
      {
      this->LoadVolumeButton->Delete ( );
      }
    if (this->VolumeSelectorWidget)
      {
      this->VolumeSelectorWidget->Delete();
      }
    if (this->WindowLevelThresholdEditor)
      {
      this->WindowLevelThresholdEditor->Delete();
      }

    this->SetModuleLogic ( NULL );
    this->SetMRMLNode ( NULL );
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
    this->LoadVolumeButton->RemoveObservers ( vtkCommand::ModifiedEvent,  (vtkCommand *)this->GUICallbackCommand );
}


//---------------------------------------------------------------------------
void vtkSlicerVolumesGUI::AddGUIObservers ( )
{

    // Fill in
    // observer load volume button
    this->LoadVolumeButton->AddObserver ( vtkCommand::ModifiedEvent, (vtkCommand *)this->GUICallbackCommand );
}


//---------------------------------------------------------------------------
void vtkSlicerVolumesGUI::ProcessGUIEvents ( vtkObject *caller,
                                             unsigned long event, void *callData )
{

    // Find the widget, and process the events from it...
    vtkKWLoadSaveButton *filebrowse = vtkKWLoadSaveButton::SafeDownCast(caller);
    if (filebrowse == this->LoadVolumeButton  && event == vtkCommand::ModifiedEvent )
        {
            // If a file has been selected for loading...
            char *fileName = filebrowse->GetFileName();
            if ( fileName ) 
                {
                    vtkSlicerVolumesLogic* volumeLogic = this->Logic;
      
                    vtkMRMLVolumeNode *volumeNode = volumeLogic->AddArchetypeVolume( fileName );
                    if ( volumeNode == NULL ) 
                        {
                        // TODO: generate an error...
                        }
                        else
                        {
                        this->ApplicationLogic->GetSelectionNode()->SetActiveVolumeID( volumeNode->GetID() );
                        this->ApplicationLogic->PropagateVolumeSelection();
                        }
                }
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
    // Fill in *placeholder GUI*
    
    vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();

    // ---
    // MODULE GUI FRAME 
    // configure a page for a volume loading UI for now.
    // later, switch on the modulesButton in the SlicerControlGUI
    // ---
    // create a page
    this->UIPanel->AddPage ( "Volumes", "Volumes", NULL );
    
    // HELP FRAME
    vtkKWFrameWithLabel *volHelpFrame = vtkKWFrameWithLabel::New ( );
    volHelpFrame->SetParent ( this->UIPanel->GetPageWidget ( "Volumes" ) );
    volHelpFrame->Create ( );
    volHelpFrame->CollapseFrame ( );
    volHelpFrame->SetLabelText ("Help");
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                  volHelpFrame->GetWidgetName(), this->UIPanel->GetPageWidget("Volumes")->GetWidgetName());

    // ---
    // LOAD FRAME            
    vtkKWFrameWithLabel *volLoadFrame = vtkKWFrameWithLabel::New ( );
    volLoadFrame->SetParent ( this->UIPanel->GetPageWidget ( "Volumes" ) );
    volLoadFrame->Create ( );
    volLoadFrame->SetLabelText ("Load");
    volLoadFrame->ExpandFrame ( );
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                  volLoadFrame->GetWidgetName(), this->UIPanel->GetPageWidget("Volumes")->GetWidgetName());
    // add a file browser 
    this->LoadVolumeButton = vtkKWLoadSaveButton::New ( );
    this->LoadVolumeButton->SetParent ( volLoadFrame->GetFrame() );
    this->LoadVolumeButton->Create ( );
    this->LoadVolumeButton->SetText ("Choose a file to load");
    this->LoadVolumeButton->GetLoadSaveDialog()->SetFileTypes(
                                                              "{ {volume} {*.*} }");

    app->Script("pack %s -side top -anchor w -padx 2 -pady 4", 
                this->LoadVolumeButton->GetWidgetName());

    // ---
    // DISPLAY FRAME            
    vtkKWFrameWithLabel *volDisplayFrame = vtkKWFrameWithLabel::New ( );
    volDisplayFrame->SetParent ( this->UIPanel->GetPageWidget ( "Volumes" ) );
    volDisplayFrame->Create ( );
    volDisplayFrame->SetLabelText ("Display");
    volDisplayFrame->SetDefaultLabelFontWeightToNormal( );
    volDisplayFrame->CollapseFrame ( );
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                  volDisplayFrame->GetWidgetName(), this->UIPanel->GetPageWidget("Volumes")->GetWidgetName());

    this->VolumeSelectorWidget = vtkSlicerNodeSelectorWidget::New() ;
    this->VolumeSelectorWidget->SetParent ( volDisplayFrame->GetFrame() );
    this->VolumeSelectorWidget->Create ( );
    this->VolumeSelectorWidget->SetNodeClass("vtkMRMLVolumeNode");
    this->VolumeSelectorWidget->SetMRMLScene(this->GetMRMLScene());
    this->VolumeSelectorWidget->UpdateMenu();
    this->VolumeSelectorWidget->SetBorderWidth(2);
    this->VolumeSelectorWidget->SetReliefToGroove();
    this->VolumeSelectorWidget->SetPadX(2);
    this->VolumeSelectorWidget->SetPadY(2);
    this->VolumeSelectorWidget->GetWidget()->GetWidget()->IndicatorVisibilityOff();
    this->VolumeSelectorWidget->GetWidget()->GetWidget()->SetWidth(24);
    this->VolumeSelectorWidget->SetLabelText( "Volume Select: ");
    this->VolumeSelectorWidget->SetBalloonHelpString("select a volume from the current mrml scene.");
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
                  this->VolumeSelectorWidget->GetWidgetName());

    this->WindowLevelThresholdEditor = vtkKWWindowLevelThresholdEditor::New();
    this->WindowLevelThresholdEditor->SetParent ( volDisplayFrame->GetFrame() );
    this->WindowLevelThresholdEditor->Create ( );
    vtkMRMLVolumeNode* volNode = NULL;
    if (this->SelectedVolumeID != NULL)
      {
      volNode =  dynamic_cast<vtkMRMLVolumeNode *> (this->GetMRMLScene()->GetNodeByID(this->SelectedVolumeID));
      }
    if (volNode != NULL)
      {
      this->WindowLevelThresholdEditor->SetImageData(volNode->GetImageData());
      }
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
                  this->WindowLevelThresholdEditor->GetWidgetName() );


    volLoadFrame->Delete();
    volHelpFrame->Delete();
    volDisplayFrame->Delete();
}





