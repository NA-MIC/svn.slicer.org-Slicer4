#include "vtkObject.h"
#include "vtkObjectFactory.h"
#include "vtkCommand.h"
#include "vtkKWWidget.h"
#include "vtkSlicerVolumesGUI.h"
#include "vtkSlicerApplication.h"
#include "vtkSlicerStyle.h"
#include "vtkKWFrameWithLabel.h"


//---------------------------------------------------------------------------
vtkStandardNewMacro (vtkSlicerVolumesGUI );
vtkCxxRevisionMacro ( vtkSlicerVolumesGUI, "$Revision: 1.0 $");


//---------------------------------------------------------------------------
vtkSlicerVolumesGUI::vtkSlicerVolumesGUI ( ) {

    //this->VolumesLogic = NULL;
    this->LoadVolumeButton = NULL;
}


//---------------------------------------------------------------------------
vtkSlicerVolumesGUI::~vtkSlicerVolumesGUI ( ) {

    if (this->LoadVolumeButton ) {
        this->LoadVolumeButton->Delete ( );
    }
    //this->VolumesLogic = NULL;
}


//---------------------------------------------------------------------------
void vtkSlicerVolumesGUI::RemoveGUIObservers ( ) {
}

//---------------------------------------------------------------------------
void vtkSlicerVolumesGUI::RemoveLogicObservers ( ) {
}
 
//---------------------------------------------------------------------------
void vtkSlicerVolumesGUI::RemoveMrmlObservers ( ) {
}

//---------------------------------------------------------------------------
void vtkSlicerVolumesGUI::AddGUIObservers ( ) {

    // observer load volume button
    this->LoadVolumeButton->AddObserver ( vtkCommand::ModifiedEvent,  (vtkCommand *)this->GUICommand );    

}


//---------------------------------------------------------------------------
void vtkSlicerVolumesGUI::AddMrmlObservers ( ) {
    
}


//---------------------------------------------------------------------------
void vtkSlicerVolumesGUI::AddLogicObservers ( ) {


}




//---------------------------------------------------------------------------
void vtkSlicerVolumesGUI::ProcessGUIEvents ( vtkObject *caller,
                                                     unsigned long event,
                                                     void *callData ) 
{
    vtkKWLoadSaveButton *filebrowse = vtkKWLoadSaveButton::SafeDownCast(caller);
    if (filebrowse == this->LoadVolumeButton  && event == vtkCommand::ModifiedEvent )
        {
            // If a file has been selected for loading...
            if ( this->LoadVolumeButton->GetFileName ( ) ) {
                this->Logic->Connect ( filebrowse->GetFileName ( ) );
            }
        }

}



//---------------------------------------------------------------------------
void vtkSlicerVolumesGUI::ProcessLogicEvents ( vtkObject *caller,
                                                     unsigned long event,
                                                     void *callData ) {

}



//---------------------------------------------------------------------------
void vtkSlicerVolumesGUI::ProcessMrmlEvents ( vtkObject *caller,
                                                     unsigned long event,
                                                     void *callData ) {

}



//---------------------------------------------------------------------------
void vtkSlicerVolumesGUI::BuildGUI ( ) {
}


//---------------------------------------------------------------------------
void vtkSlicerVolumesGUI::BuildGUI ( vtkKWWidget* f ) {

    vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
    vtkSlicerStyle *style = app->GetSlicerStyle();
    // ---
    // MODULE GUI FRAME 
    // configure a page for a volume loading UI for now.
    // later, switch on the modulesButton in the SlicerControlGUI
    // ---
    // HELP FRAME
    vtkKWFrameWithLabel *volHelpFrame = vtkKWFrameWithLabel::New ( );
    volHelpFrame->SetParent ( f );
    volHelpFrame->Create ( );
    volHelpFrame->CollapseFrame ( );
    volHelpFrame->SetLabelText ("Help");
    volHelpFrame->SetDefaultLabelFontWeightToNormal( );
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                  volHelpFrame->GetWidgetName(), f->GetWidgetName());

    // ---
    // LOAD FRAME            
    vtkKWFrameWithLabel *volLoadFrame = vtkKWFrameWithLabel::New ( );
    volLoadFrame->SetParent ( f );
    volLoadFrame->Create ( );
    volLoadFrame->SetLabelText ("Load");
    volLoadFrame->SetDefaultLabelFontWeightToNormal( );
    volLoadFrame->ExpandFrame ( );

    // add a file browser 
    this->LoadVolumeButton = vtkKWLoadSaveButton::New ( );
    this->LoadVolumeButton->SetParent ( volLoadFrame->GetFrame() );
    this->LoadVolumeButton->Create ( );
    this->LoadVolumeButton->SetText ("Choose a file to load");
    this->LoadVolumeButton->GetLoadSaveDialog()->SetFileTypes(
                                                              "{ {MRML Document} {.mrml .xml} }");
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                  volLoadFrame->GetWidgetName(), f->GetWidgetName());
    app->Script("pack %s -side top -anchor w -padx 2 -pady 4", 
                this->LoadVolumeButton->GetWidgetName());

    // ---
    // DISPLAY FRAME            
    vtkKWFrameWithLabel *volDisplayFrame = vtkKWFrameWithLabel::New ( );
    volDisplayFrame->SetParent ( f );
    volDisplayFrame->Create ( );
    volDisplayFrame->SetLabelText ("Display");
    volDisplayFrame->SetDefaultLabelFontWeightToNormal( );
    volDisplayFrame->CollapseFrame ( );
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                  volDisplayFrame->GetWidgetName(), f->GetWidgetName());

}





