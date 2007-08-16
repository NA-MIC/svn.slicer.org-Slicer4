#include "vtkObjectFactory.h"
#include "vtkCommand.h"
#include "vtkImageData.h"

#include "vtkSlicerSliceControllerWidget.h"
#include "vtkSlicerApplication.h"
#include "vtkSlicerSlicesGUI.h"
#include "vtkSlicerApplicationGUI.h"
#include "vtkSlicerSlicesControlGUI.h"
#include "vtkSlicerVolumesGUI.h"
#include "vtkSlicerTheme.h"

#include "vtkKWWidget.h"
#include "vtkKWScaleWithEntry.h"
#include "vtkKWEntryWithLabel.h"
#include "vtkKWEntry.h"
#include "vtkKWScale.h"
#include "vtkKWLabel.h"
#include "vtkKWRenderWidget.h"
#include "vtkKWMenu.h"
#include "vtkKWMenuButton.h"
#include "vtkKWPushButton.h"
#include "vtkKWTkUtilities.h"
#include "vtkKWIcon.h"


//---------------------------------------------------------------------------
vtkStandardNewMacro ( vtkSlicerSliceControllerWidget );
vtkCxxRevisionMacro ( vtkSlicerSliceControllerWidget, "$Revision: 1.0 $");


//---------------------------------------------------------------------------
vtkSlicerSliceControllerWidget::vtkSlicerSliceControllerWidget ( ) {

  //---  
  // widgets comprising the SliceControllerWidget for now.
  this->OffsetScale = NULL;
  this->OrientationSelector = NULL;
  this->ForegroundSelector = NULL;
  this->BackgroundSelector = NULL;
  this->LabelSelector = NULL;
  this->OrientationMenuButton  = NULL;
  this->ForegroundMenuButton = NULL;
  this->BackgroundMenuButton = NULL;
  this->LabelMenuButton = NULL;
  this->VisibilityToggle = NULL;
  this->LabelOpacityButton = NULL;
  this->LabelOpacityScale = NULL;
  this->LabelOpacityTopLevel = NULL;
  this->LightboxTopLevel = NULL;
  this->LinkButton = NULL;
  this->VisibilityIcons = NULL;
  this->ViewConfigureIcons = NULL;
  this->SliceNode = NULL;
  this->SliceCompositeNode = NULL;
  this->SliceLogic = NULL;
  this->ScaleFrame = NULL;
  this->ColorCodeButton = NULL;
  this->SliceControlIcons = NULL;
  this->ContainerFrame = NULL;
  this->FitToWindowButton = NULL;
  this->VolumeDisplayMenuButton = NULL;
  this->LightboxButton = NULL;
  this->LightboxWidthEntry = NULL;
  this->LightboxHeightEntry = NULL;
  this->LightboxApplyButton = NULL;
}


//---------------------------------------------------------------------------
vtkSlicerSliceControllerWidget::~vtkSlicerSliceControllerWidget ( ){

  if ( this->FitToWindowButton )
    {
    this->FitToWindowButton->SetParent ( NULL );
    this->FitToWindowButton->Delete( );
    this->FitToWindowButton = NULL;
    }
  if ( this->OffsetScale )
    {
    this->OffsetScale->SetParent(NULL);
    this->OffsetScale->Delete ( );
    this->OffsetScale = NULL;
    }
  if ( this->OrientationSelector )
    {
    this->OrientationSelector->SetParent(NULL);
    this->OrientationSelector->Delete ( );
    this->OrientationSelector = NULL;
    }
  if ( this->ForegroundSelector )
    {
    this->ForegroundSelector->SetParent(NULL);
    this->ForegroundSelector->Delete ( );
    this->ForegroundSelector = NULL;
    }
  if ( this->BackgroundSelector )
    {
    this->BackgroundSelector->SetParent(NULL);
    this->BackgroundSelector->Delete ( );
    this->BackgroundSelector = NULL;
    }
  if ( this->LabelSelector )
    {
    this->LabelSelector->SetParent(NULL);
    this->LabelSelector->Delete ( );
    this->LabelSelector = NULL;
    }
  if ( this->OrientationMenuButton )
    {
    this->OrientationMenuButton->SetParent ( NULL );
    this->OrientationMenuButton->Delete();
    this->OrientationMenuButton = NULL;
    }
  if ( this->ForegroundMenuButton )
    {
    this->ForegroundMenuButton->SetParent ( NULL );
    this->ForegroundMenuButton->Delete();
    this->ForegroundMenuButton = NULL;    
    }
  if ( this->BackgroundMenuButton )
    {
    this->BackgroundMenuButton->SetParent ( NULL );
    this->BackgroundMenuButton->Delete();
    this->BackgroundMenuButton = NULL;    
    }
  if ( this->LabelMenuButton )
    {
    this->LabelMenuButton->SetParent ( NULL );
    this->LabelMenuButton->Delete();
    this->LabelMenuButton = NULL;    
    }
  if ( this->VolumeDisplayMenuButton)
    {
    this->VolumeDisplayMenuButton->SetParent(NULL);
    this->VolumeDisplayMenuButton->Delete  ( );
    this->VolumeDisplayMenuButton = NULL;
    }
  if ( this->VisibilityToggle )
    {
    this->VisibilityToggle->SetParent(NULL);
    this->VisibilityToggle->Delete  ( );
    this->VisibilityToggle = NULL;
    }
  if ( this->LabelOpacityButton )
    {
    this->LabelOpacityButton->SetParent(NULL);
    this->LabelOpacityButton->Delete  ( );
    this->LabelOpacityButton = NULL;
    }
  if ( this->LabelOpacityScale )
    {
    this->LabelOpacityScale->SetParent(NULL);
    this->LabelOpacityScale->Delete  ( );
    this->LabelOpacityScale = NULL;
    }
  if ( this->LabelOpacityTopLevel )
    {
    this->LabelOpacityTopLevel->SetParent(NULL);
    this->LabelOpacityTopLevel->Delete  ( );
    this->LabelOpacityTopLevel = NULL;
    }
  if ( this->LightboxTopLevel )
    {
    this->LightboxTopLevel->SetParent(NULL);
    this->LightboxTopLevel->Delete  ( );
    this->LightboxTopLevel = NULL;
    }
  if ( this->LightboxButton )
    {
    this->LightboxButton->SetParent ( NULL );
    this->LightboxButton->Delete();
    this->LightboxButton = NULL;
    }
  if ( this->LightboxWidthEntry )
    {
    this->LightboxWidthEntry->SetParent ( NULL );
    this->LightboxWidthEntry->Delete();
    this->LightboxWidthEntry = NULL;    
    }
  if ( this->LightboxHeightEntry )
    {
    this->LightboxHeightEntry->SetParent ( NULL );
    this->LightboxHeightEntry->Delete();
    this->LightboxHeightEntry = NULL;    
    }
  if ( this->LinkButton )
    {
    this->LinkButton->SetParent(NULL);
    this->LinkButton->Delete  ( );
    this->LinkButton = NULL;
    }
  if ( this->VisibilityIcons )
    {
    this->VisibilityIcons->Delete  ( );
    this->VisibilityIcons = NULL;
    }
  if ( this->ViewConfigureIcons )
    {
    this->ViewConfigureIcons->Delete ( );
    this->ViewConfigureIcons = NULL;
    }
  if ( this->SliceControlIcons )
    {
    this->SliceControlIcons->Delete  ( );
    this->SliceControlIcons = NULL;
    }
  if ( this->ScaleFrame )
    {
    this->ScaleFrame->SetParent(NULL);
    this->ScaleFrame->Delete ( );
    this->ScaleFrame = NULL;
    }
  if ( this->ColorCodeButton )
    {
    this->ColorCodeButton->SetParent(NULL);
    this->ColorCodeButton->Delete ( );
    this->ColorCodeButton = NULL;
    }
  if ( this->ContainerFrame )
    {
    this->ContainerFrame->SetParent(NULL);
    this->ContainerFrame->Delete ( );
    this->ContainerFrame = NULL;
    }
  if ( this->LightboxButton )
    {
    this->LightboxButton->SetParent ( NULL );
    this->LightboxButton->Delete();
    this->LightboxButton = NULL;
    }
  if ( this->LightboxApplyButton )
    {
    this->LightboxApplyButton->SetParent ( NULL );
    this->LightboxApplyButton->Delete();
    this->LightboxApplyButton = NULL;
    }
  this->SetSliceNode ( NULL );
  this->SetSliceCompositeNode ( NULL );
  this->SetSliceLogic ( NULL );
}




//----------------------------------------------------------------------------
void vtkSlicerSliceControllerWidget::AddWidgetObservers ( )
{
  if ( this->OffsetScale == NULL ) 
    {
    vtkErrorMacro ("Can't add observers because CreateWidget hasn't been called");
    return;
    }

    this->OrientationSelector->GetWidget()->GetWidget()->GetMenu()->AddObserver ( vtkKWMenu::MenuItemInvokedEvent, this->GUICallbackCommand);
    this->ForegroundSelector->AddObserver ( vtkSlicerNodeSelectorWidget::NodeSelectedEvent, this->GUICallbackCommand);
    this->BackgroundSelector->AddObserver ( vtkSlicerNodeSelectorWidget::NodeSelectedEvent, this->GUICallbackCommand);
    this->LabelSelector->AddObserver ( vtkSlicerNodeSelectorWidget::NodeSelectedEvent, this->GUICallbackCommand);
    this->OffsetScale->GetWidget()->AddObserver( vtkKWScale::ScaleValueChangingEvent, this->GUICallbackCommand );
    this->OffsetScale->GetWidget()->AddObserver( vtkKWScale::ScaleValueChangedEvent, this->GUICallbackCommand );
    this->OffsetScale->GetWidget()->AddObserver( vtkKWScale::ScaleValueStartChangingEvent, this->GUICallbackCommand );
    this->VisibilityToggle->AddObserver (vtkKWPushButton::InvokedEvent, this->GUICallbackCommand );
    this->LabelOpacityButton->AddObserver (vtkKWPushButton::InvokedEvent, this->GUICallbackCommand );
    this->LabelOpacityScale->GetScale ( )->AddObserver( vtkKWScale::ScaleValueStartChangingEvent, this->GUICallbackCommand );
    this->LabelOpacityScale->GetScale ( )->AddObserver( vtkKWScale::ScaleValueChangingEvent, this->GUICallbackCommand );
    this->LabelOpacityScale->GetScale ( )->AddObserver( vtkKWScale::ScaleValueChangedEvent, this->GUICallbackCommand );    
    this->LinkButton->AddObserver (vtkKWPushButton::InvokedEvent, this->GUICallbackCommand );
    this->FitToWindowButton->AddObserver (vtkKWPushButton::InvokedEvent, this->GUICallbackCommand );
    this->VolumeDisplayMenuButton->GetMenu()->AddObserver (vtkKWMenu::MenuItemInvokedEvent, this->GUICallbackCommand );    
    this->LightboxButton->GetMenu()->AddObserver ( vtkKWMenu::MenuItemInvokedEvent, this->GUICallbackCommand );
    this->LightboxApplyButton->AddObserver (vtkKWPushButton::InvokedEvent, this->GUICallbackCommand );
    this->ForegroundMenuButton->GetMenu()->AddObserver ( vtkKWMenu::MenuItemInvokedEvent, this->GUICallbackCommand );
    this->BackgroundMenuButton->GetMenu()->AddObserver ( vtkKWMenu::MenuItemInvokedEvent, this->GUICallbackCommand );
}
  

//---------------------------------------------------------------------------
void vtkSlicerSliceControllerWidget::RemoveWidgetObservers ( ) {

  if ( this->OffsetScale == NULL ) 
    {
    vtkErrorMacro ("Can't remove observers because CreateWidget hasn't been called");
    return;
    }

    this->OrientationSelector->GetWidget()->GetWidget()->GetMenu()->RemoveObservers ( vtkKWMenu::MenuItemInvokedEvent, this->GUICallbackCommand);
    this->ForegroundSelector->RemoveObservers ( vtkSlicerNodeSelectorWidget::NodeSelectedEvent, this->GUICallbackCommand);
    this->BackgroundSelector->RemoveObservers ( vtkSlicerNodeSelectorWidget::NodeSelectedEvent, this->GUICallbackCommand);
    this->LabelSelector->RemoveObservers ( vtkSlicerNodeSelectorWidget::NodeSelectedEvent, this->GUICallbackCommand);
    this->LabelOpacityScale->GetScale ( )->RemoveObservers( vtkKWScale::ScaleValueStartChangingEvent, this->GUICallbackCommand );
    this->LabelOpacityScale->GetScale ( )->RemoveObservers( vtkKWScale::ScaleValueChangingEvent, this->GUICallbackCommand );
    this->LabelOpacityScale->GetScale ( )->RemoveObservers( vtkKWScale::ScaleValueChangedEvent, this->GUICallbackCommand );    
    this->OffsetScale->GetWidget()->RemoveObservers ( vtkKWScale::ScaleValueChangingEvent, this->GUICallbackCommand );
    this->OffsetScale->GetWidget()->RemoveObservers ( vtkKWScale::ScaleValueChangedEvent, this->GUICallbackCommand );
    this->OffsetScale->GetWidget()->RemoveObservers ( vtkKWScale::ScaleValueStartChangingEvent, this->GUICallbackCommand );
    this->VisibilityToggle->RemoveObservers ( vtkKWPushButton::InvokedEvent, this->GUICallbackCommand );
    this->LabelOpacityButton->RemoveObservers ( vtkKWPushButton::InvokedEvent, this->GUICallbackCommand );
    this->LinkButton->RemoveObservers ( vtkKWPushButton::InvokedEvent, this->GUICallbackCommand );        
    this->FitToWindowButton->RemoveObservers (vtkKWPushButton::InvokedEvent, this->GUICallbackCommand );
    this->VolumeDisplayMenuButton->GetMenu()->RemoveObservers (vtkKWMenu::MenuItemInvokedEvent, this->GUICallbackCommand );
    this->LightboxButton->GetMenu()->RemoveObservers ( vtkKWMenu::MenuItemInvokedEvent, this->GUICallbackCommand );
    this->LightboxApplyButton->RemoveObservers (vtkKWPushButton::InvokedEvent, this->GUICallbackCommand );
    this->ForegroundMenuButton->GetMenu()->RemoveObservers ( vtkKWMenu::MenuItemInvokedEvent, this->GUICallbackCommand );
    this->BackgroundMenuButton->GetMenu()->RemoveObservers ( vtkKWMenu::MenuItemInvokedEvent, this->GUICallbackCommand );
}



//---------------------------------------------------------------------------
void vtkSlicerSliceControllerWidget::ApplyColorCode ( double *c )
{
  this->ColorCodeButton->SetBackgroundColor (c[0], c[1], c[2] );
}



//---------------------------------------------------------------------------
void vtkSlicerSliceControllerWidget::CreateWidget ( ) 
{

    if ( !this->MRMLScene ) {
        vtkErrorMacro ( << " MRML Scene must be set before creating widgets.");
        return;
    }

    // the widget is a frame with some widgets inside
    if (this->IsCreated ( ) ) {
        vtkErrorMacro ( << this->GetClassName() << "already created.");
        return;
        
    }
    this->Superclass::CreateWidget ( );
    this->SliceControlIcons = vtkSlicerSlicesControlIcons::New ( );
    //
    // A stripe that color codes the SliceGUI this controller belongs to.
    //

    this->ColorCodeButton = vtkKWPushButton::New ( );
    this->ColorCodeButton->SetParent ( this );
    this->ColorCodeButton->Create ( );
    this->ColorCodeButton->SetBorderWidth (0 );
    this->ColorCodeButton->SetImageToPredefinedIcon (vtkKWIcon::IconSpinUp );
    this->ColorCodeButton->SetHeight (7 );
    this->ColorCodeButton->SetCommand (this, "Shrink");
    this->ColorCodeButton->SetBalloonHelpString ("Click to shrink/expand" );

    this->ContainerFrame = vtkKWFrame::New ( );
    this->ContainerFrame->SetParent ( this );
    this->ContainerFrame->Create ( );

    vtkSlicerApplication *app = vtkSlicerApplication::SafeDownCast(this->GetApplication() );

    //
    // Foreground, Background, Label and Orientation MenuButtons + Menus
    //
    this->OrientationMenuButton = vtkKWMenuButton::New();
    this->OrientationMenuButton->SetParent ( this->ContainerFrame);
    this->OrientationMenuButton->Create ( );
    this->OrientationMenuButton->SetBorderWidth ( 0 );
    this->OrientationMenuButton->SetImageToIcon ( this->SliceControlIcons->GetSetOrIcon() );
    this->OrientationMenuButton->IndicatorVisibilityOff ( );
    this->OrientationMenuButton->SetBalloonHelpString ( "Select options for the Viewer's orientation (not yet implemented)." );    
//    this->OrientationMenuButton->GetMenu()->DeleteAllItems();
//    this->OrientationMenuButton->GetMenu()->AddRadioButton ( "..." );
//    this->OrientationMenuButton->AddSeparator ( );
//    this->OrientationMenuButton->GetMenu()->AddCommand ( "close" );

    
    this->LabelMenuButton = vtkKWMenuButton::New();
    this->LabelMenuButton->SetParent ( this->ContainerFrame);
    this->LabelMenuButton->Create ( );
    this->LabelMenuButton->SetBorderWidth ( 0 );
    this->LabelMenuButton->SetImageToIcon ( this->SliceControlIcons->GetSetLbIcon() );
    this->LabelMenuButton->IndicatorVisibilityOff ( );
    this->LabelMenuButton->SetBalloonHelpString ( "Select options for the Label Layer (not yet implemented)." );    
//    this->LabelMenuButton->GetMenu()->DeleteAllItems();
//    this->LabelMenuButton->GetMenu()->AddRadioButton ( "...");
//    this->LabelMenuButton->GetMenu()->AddSeparator ( );    
//    this->LabelMenuButton->GetMenu()->AddCommand ( "close ");
    

    this->ForegroundMenuButton = vtkKWMenuButton::New();
    this->ForegroundMenuButton->SetParent ( this->ContainerFrame);
    this->ForegroundMenuButton->Create ( );
    this->ForegroundMenuButton->SetBorderWidth ( 0 );
    this->ForegroundMenuButton->SetImageToIcon ( this->SliceControlIcons->GetSetFgIcon() );
    this->ForegroundMenuButton->IndicatorVisibilityOff ( );
    this->ForegroundMenuButton->SetBalloonHelpString ( "Select options for the Foreground Layer" );    
    this->ForegroundMenuButton->GetMenu()->DeleteAllItems ( );
    this->ForegroundMenuButton->GetMenu()->AddCheckButton ( "interpolation" );
    this->ForegroundMenuButton->GetMenu()->AddSeparator ( );
    this->ForegroundMenuButton->GetMenu()->AddCommand ( "close");    
    
    this->BackgroundMenuButton = vtkKWMenuButton::New();
    this->BackgroundMenuButton->SetParent ( this->ContainerFrame);
    this->BackgroundMenuButton->Create ( );
    this->BackgroundMenuButton->SetBorderWidth ( 0 );
    this->BackgroundMenuButton->SetImageToIcon ( this->SliceControlIcons->GetSetBgIcon() );
    this->BackgroundMenuButton->IndicatorVisibilityOff ( );
    this->BackgroundMenuButton->SetBalloonHelpString ( "Select options for the Background Layer" );    
    this->BackgroundMenuButton->GetMenu()->DeleteAllItems ( );
    this->BackgroundMenuButton->GetMenu()->AddCheckButton ( "interpolation" );
    this->BackgroundMenuButton->GetMenu()->AddSeparator ( );
    this->BackgroundMenuButton->GetMenu()->AddCommand ( "close");    

    //
    // Orientation  (TODO: make this into a vtkSlicerOrientationWidget)
    //
    this->OrientationSelector = vtkKWMenuButtonWithSpinButtonsWithLabel::New ();
    this->OrientationSelector->SetParent ( this->ContainerFrame );
    this->OrientationSelector->Create ( );    
    this->OrientationSelector->SetBalloonHelpString ("Select orientation" );
    vtkKWMenuButton *mb = this->OrientationSelector->GetWidget()->GetWidget();
    mb->SetWidth ( 10 );
    mb->GetMenu()->AddRadioButton ( "Axial" );
    mb->GetMenu()->AddRadioButton ( "Sagittal" );
    mb->GetMenu()->AddRadioButton ( "Coronal" );
    mb->SetValue ("Axial");    

    // Foreground, Background, and Label selections
    //
    this->ForegroundSelector = vtkSlicerNodeSelectorWidget::New();
   this->ForegroundSelector->SetParent ( this->ContainerFrame );
    this->ForegroundSelector->Create ( );
    this->ForegroundSelector->NoneEnabledOn();
    this->ForegroundSelector->SetBalloonHelpString ( "Select the foreground");
    this->ForegroundSelector->SetNodeClass ("vtkMRMLVolumeNode", NULL, NULL, NULL);
    this->ForegroundSelector->SetMRMLScene( this->MRMLScene );
    this->ForegroundSelector->GetWidget()->GetWidget()->SetMaximumLabelWidth(10);
    this->ForegroundSelector->GetWidget()->GetWidget()->SetWidth(10);

    this->BackgroundSelector = vtkSlicerNodeSelectorWidget::New();
    this->BackgroundSelector->SetParent ( this->ContainerFrame );
    this->BackgroundSelector->Create ( );
    this->BackgroundSelector->NoneEnabledOn();
    
    this->BackgroundSelector->SetBalloonHelpString ( "Select the background");
    this->BackgroundSelector->SetNodeClass ("vtkMRMLVolumeNode", NULL, NULL, NULL);
    this->BackgroundSelector->SetMRMLScene( this->MRMLScene );
    this->BackgroundSelector->GetWidget()->GetWidget()->SetMaximumLabelWidth(10);
    this->BackgroundSelector->GetWidget()->GetWidget()->SetWidth(10);

    this->LabelSelector = vtkSlicerNodeSelectorWidget::New();
    this->LabelSelector->SetParent ( this->ContainerFrame );
    this->LabelSelector->Create ( );
    this->LabelSelector->NoneEnabledOn();
    this->LabelSelector->SetBalloonHelpString ( "Select the label map");
    this->LabelSelector->SetNodeClass ("vtkMRMLVolumeNode", "LabelMap", "1", NULL);
    this->LabelSelector->SetMRMLScene( this->MRMLScene );
    this->LabelSelector->GetWidget()->GetWidget()->SetMaximumLabelWidth(10);
    this->LabelSelector->GetWidget()->GetWidget()->SetWidth(10);

    //
    // Create the frame to contain scale and visibility toggle
    //
    this->ScaleFrame = vtkKWFrame::New ();
    this->ScaleFrame->SetParent ( this->ContainerFrame );
    this->ScaleFrame->Create ( );

    //
    // Create a button to toggle the slice visibility in the main viewer and icons for it
    //
    this->VisibilityIcons = vtkSlicerVisibilityIcons::New ( );
    this->VisibilityToggle = vtkKWPushButton::New ( );
    this->VisibilityToggle->SetParent ( this->ScaleFrame );
    this->VisibilityToggle->Create ( );
    this->VisibilityToggle->SetReliefToFlat ( );
    this->VisibilityToggle->SetOverReliefToNone ( );
    this->VisibilityToggle->SetBorderWidth ( 0 );
    this->VisibilityToggle->SetImageToIcon ( this->VisibilityIcons->GetInvisibleIcon ( ) );        
    this->VisibilityToggle->SetBalloonHelpString ( "Toggles slice visibility in the MainViewer." );

    //
    // Create a button to toggle the slice visibility in the main viewer and icons for it
    //
    this->LinkButton = vtkKWPushButton::New ( );
    this->LinkButton->SetParent ( this->ScaleFrame );
    this->LinkButton->Create ( );
    this->LinkButton->SetReliefToFlat ( );
    this->LinkButton->SetOverReliefToNone ( );
    this->LinkButton->SetBorderWidth ( 0 );
    this->LinkButton->SetImageToIcon ( this->SliceControlIcons->GetUnlinkControlsIcon ( ) );        
    this->LinkButton->SetBalloonHelpString ( "Links/Unlinks the slice controls (except scales) across all Slice Viewers." );

    //
    // Create a button to fit the view to the window
    //
    this->FitToWindowButton = vtkKWPushButton::New ( );
    this->FitToWindowButton->SetParent ( this->ScaleFrame );
    this->FitToWindowButton->Create ( );
    this->FitToWindowButton->SetReliefToFlat ( );
    this->FitToWindowButton->SetOverReliefToNone ( );
    this->FitToWindowButton->SetBorderWidth ( 0 );
    this->FitToWindowButton->SetImageToIcon ( this->SliceControlIcons->GetFitToWindowIcon ( ));    
    this->FitToWindowButton->SetBalloonHelpString ( "Adjusts the Slice Viewer's field of view to match the extent of current background volume.");
// adjust the node's field of view to match the extent of current background volume

    //
    // Create a menubutton that navigates to Volumes->Display
    // 
    this->VolumeDisplayMenuButton = vtkKWMenuButton::New ( );
    this->VolumeDisplayMenuButton->SetParent ( this->ScaleFrame );
    this->VolumeDisplayMenuButton->Create ( );
    this->VolumeDisplayMenuButton->SetBorderWidth ( 0 );
    this->VolumeDisplayMenuButton->IndicatorVisibilityOff ( );
    this->VolumeDisplayMenuButton->SetImageToIcon ( this->SliceControlIcons->GetWinLevThreshColIcon ( ));    
    this->VolumeDisplayMenuButton->SetBalloonHelpString ( "Adjust window, level, threshold and color palette for a Slice Layer.");
    this->VolumeDisplayMenuButton->GetMenu()->AddRadioButton ( "Foreground volume" );
    this->VolumeDisplayMenuButton->GetMenu()->AddRadioButton ( "Background volume" );
    this->VolumeDisplayMenuButton->GetMenu()->AddRadioButton ( "Label map" );
//    this->VolumeDisplayMenuButton->GetMenu()->SetItemStateToDisabled ("Foreground volume");
//    this->VolumeDisplayMenuButton->GetMenu()->SetItemStateToDisabled ("Background volume");
//    this->VolumeDisplayMenuButton->GetMenu()->SetItemStateToDisabled ("Label map");
    this->VolumeDisplayMenuButton->GetMenu()->AddSeparator();
    this->VolumeDisplayMenuButton->GetMenu()->AddCommand ( "close" );

    //
    // Create a lightbox menubutton that allows viewer to be reconfigured
    //
    this->ViewConfigureIcons = vtkSlicerToolbarIcons::New ( );
    this->LightboxButton = vtkKWMenuButton::New();
    this->LightboxButton->SetParent ( this->ScaleFrame );
    this->LightboxButton->Create();
    this->LightboxButton->SetBorderWidth ( 0 );
    this->LightboxButton->IndicatorVisibilityOff ( );
    this->LightboxButton->SetImageToIcon ( this->ViewConfigureIcons->GetLightBoxViewIcon ( ) );
    this->LightboxButton->SetBalloonHelpString ( "Configure the Slice viewer layout");
    this->LightboxButton->GetMenu()->AddRadioButton ("1x1 view");
    this->LightboxButton->GetMenu()->AddRadioButton ("2x2 view");
    this->LightboxButton->GetMenu()->AddRadioButton ("3x3 view");
    this->LightboxButton->GetMenu()->AddRadioButton ( "6x6 view");    
    this->LightboxButton->GetMenu()->AddRadioButton ( "customized view");    
    this->LightboxButton->GetMenu()->AddSeparator ( );
    this->LightboxButton->GetMenu()->AddCommand ("close");
    this->LightboxButton->GetMenu()->SetItemStateToDisabled ( "customized view" );
            
    //--- Pop-up frame for custom NXM lightbox configuration
    this->LightboxTopLevel = vtkKWTopLevel::New ( );
    this->LightboxTopLevel->SetApplication ( app );
    this->LightboxTopLevel->SetMasterWindow ( this->LightboxButton );
    this->LightboxTopLevel->Create ( );
    this->LightboxTopLevel->HideDecorationOn ( );
    this->LightboxTopLevel->Withdraw ( );
    this->LightboxTopLevel->SetBorderWidth ( 2 );
    this->LightboxTopLevel->SetReliefToGroove ( );

    //--- create temporary pop-up frame to display when custom configuration is selected
    vtkKWFrame *popUpFrame1 = vtkKWFrame::New ( );
    popUpFrame1->SetParent ( this->LightboxTopLevel );
    popUpFrame1->Create ( );
    popUpFrame1->SetBinding ( "<Leave>", this, "HideLightboxCustomLayoutFrame" );
    this->Script ( "pack %s -side left -anchor w -padx 2 -pady 2 -fill x -fill y -expand n", popUpFrame1->GetWidgetName ( ) );   
    this->LightboxWidthEntry = vtkKWEntry::New ( );
    this->LightboxWidthEntry->SetParent ( popUpFrame1 );
    this->LightboxWidthEntry->Create ( );
    this->LightboxWidthEntry->SetValueAsInt (1);
    this->LightboxWidthEntry->SetWidth ( 3 );
    this->LightboxHeightEntry = vtkKWEntry::New ( );
    this->LightboxHeightEntry->SetParent ( popUpFrame1 );
    this->LightboxHeightEntry->Create ( );
    this->LightboxHeightEntry->SetWidth ( 3 );
    this->LightboxHeightEntry->SetValueAsInt (1);
    vtkKWLabel *widthLabel = vtkKWLabel::New();
    widthLabel->SetParent ( popUpFrame1 );
    widthLabel->Create ( );
    widthLabel->SetText ( "horizontal:" );
    vtkKWLabel *heightLabel = vtkKWLabel::New();
    heightLabel->SetParent ( popUpFrame1 );
    heightLabel->Create ( );
    heightLabel->SetText ( "vertical:" );
    this->LightboxApplyButton = vtkKWPushButton::New ( );
    this->LightboxApplyButton->SetParent ( popUpFrame1 );
    this->LightboxApplyButton->Create ( );
    this->LightboxApplyButton->SetText ("Apply");    
    this->Script ( "grid %s -row 0 -column 0 -padx 2 -pady 8", widthLabel->GetWidgetName());
    this->Script ( "grid %s -row 0 -column 1 -padx 6 -pady 8", this->LightboxWidthEntry->GetWidgetName() );
    this->Script ( "grid %s -row 1 -column 0 -padx 2 -pady 8", heightLabel->GetWidgetName());
    this->Script ( "grid %s -row 1 -column 1 -padx 6 -pady 8", this->LightboxHeightEntry->GetWidgetName() );
    this->Script ( "grid %s -row 2 -column 0 -columnspan 2 -pady 8", this->LightboxApplyButton->GetWidgetName() );
    // delete temporary stuff
    widthLabel->Delete();
    heightLabel->Delete();
    popUpFrame1->Delete();
    
    //--- Popup Scale with Entry (displayed when user clicks LabelOpacityButton
    //--- LabelOpacityButton, LabelOpacityScale and its entry will be observed
    //--- and their events handled in ProcessGUIEvents;
    //--- the pop-up and hide behavior of the latter two will be managed locally
    //--- in the GUI.
    //--- TODO: make a SlicerWidget that handles this behavior. Leave event?
    this->LabelOpacityTopLevel = vtkKWTopLevel::New ( );
    this->LabelOpacityTopLevel->SetApplication ( app );
    this->LabelOpacityTopLevel->SetMasterWindow ( this->LabelOpacityButton );
    this->LabelOpacityTopLevel->Create ( );
    this->LabelOpacityTopLevel->HideDecorationOn ( );
    this->LabelOpacityTopLevel->Withdraw ( );
    this->LabelOpacityTopLevel->SetBorderWidth ( 2 );
    this->LabelOpacityTopLevel->SetReliefToGroove ( );
    //--- create temporary frame
    vtkKWFrame *popUpFrame2 = vtkKWFrame::New ( );
    popUpFrame2->SetParent ( this->LabelOpacityTopLevel );
    popUpFrame2->Create ( );
    popUpFrame2->SetBinding ( "<Leave>", this, "HideLabelOpacityScaleAndEntry" );
    this->Script ( "pack %s -side left -anchor w -padx 2 -pady 2 -fill x -fill y -expand n", popUpFrame2->GetWidgetName ( ) );   
    // Scale and entry packed in the pop-up toplevel's frame
    this->LabelOpacityScale = vtkKWScaleWithEntry::New ( );
    this->LabelOpacityScale->SetParent ( popUpFrame2 );
    this->LabelOpacityScale->Create ( );
    this->LabelOpacityScale->SetRange ( 0.0, 1.0 );
    this->LabelOpacityScale->SetResolution ( 0.01 );
    this->LabelOpacityScale->GetScale()->SetLabelText ( "" );
    this->LabelOpacityScale->GetScale()->ValueVisibilityOff ( );
    this->LabelOpacityScale->SetValue ( 1.0 );
    this->Script ( "pack %s -side left -anchor w -padx 1 -pady 3 -expand n", this->LabelOpacityScale->GetWidgetName ( ) );
    this->LabelOpacityButton = vtkKWPushButton::New ( );
    this->LabelOpacityButton->SetParent (this->ScaleFrame );
    this->LabelOpacityButton->Create ( );
    this->LabelOpacityButton->SetBorderWidth ( 0 );
    this->LabelOpacityButton->SetImageToIcon ( this->SliceControlIcons->GetLabelOpacityIcon() );
    this->LabelOpacityButton->SetBalloonHelpString ( "Popup scale to adjust opacity of Label Layer." );
    //--- delete temporary frame
    popUpFrame2->Delete ( );

    //
    // Create a scale to control the slice number displayed
    //
    this->OffsetScale = vtkKWScaleWithEntry::New();
    this->OffsetScale->SetParent ( this->ScaleFrame );
    this->OffsetScale->Create();
    this->OffsetScale->RangeVisibilityOff ( );
    this->OffsetScale->SetEntryWidth(8);
    this->OffsetScale->SetLabelPositionToLeft();

            
    //
    // Pack everyone up
    ///
    this->Script ( "pack %s -side top -expand 1 -fill x", 
                   this->ColorCodeButton->GetWidgetName ( ));
    this->Script ("pack %s -side bottom -expand 1 -fill x", 
                  this->ContainerFrame->GetWidgetName());
    this->Script("grid columnconfigure %s 0 -weight 0", 
                 this->ContainerFrame->GetWidgetName());
    this->Script("grid columnconfigure %s 1 -weight 1", 
                 this->ContainerFrame->GetWidgetName());
    this->Script("grid columnconfigure %s 2 -weight 0", 
                 this->ContainerFrame->GetWidgetName());
    this->Script("grid columnconfigure %s 3 -weight 1", 
                 this->ContainerFrame->GetWidgetName());

    
    this->Script("grid %s %s %s %s -sticky ew -padx 1",
                 this->OrientationMenuButton->GetWidgetName (),
                 this->OrientationSelector->GetWidgetName(),
                 this->ForegroundMenuButton->GetWidgetName(),
                 this->ForegroundSelector->GetWidgetName());
    this->Script("grid %s %s %s %s -sticky ew -padx 1", 
                 this->LabelMenuButton->GetWidgetName(),
                 this->LabelSelector->GetWidgetName(),
                 this->BackgroundMenuButton->GetWidgetName(),
                 this->BackgroundSelector->GetWidgetName());
    this->Script ( "grid %s -sticky ew -columnspan 4", 
                   this->ScaleFrame->GetWidgetName ( ) );

    this->Script ("pack %s -side left -expand n -padx 1", 
                  this->LinkButton->GetWidgetName ( ) );
    this->Script ("pack %s -side left -expand n -padx 1", 
                  this->VisibilityToggle->GetWidgetName ( ) );
    this->Script ("pack %s -side left -expand n -padx 1", 
                  this->FitToWindowButton->GetWidgetName ( ) );
    this->Script ("pack %s -side left -expand n -padx 1", 
                  this->LabelOpacityButton->GetWidgetName ( ) );
    this->Script ("pack %s -side left -expand n -padx 1", 
                  this->VolumeDisplayMenuButton->GetWidgetName ( ) );    
    this->Script ("pack %s -side left -expand n -padx 1", 
                  this->LightboxButton->GetWidgetName ( ) );    
    this->Script("pack %s -side left -fill x -expand y", 
                 this->OffsetScale->GetWidgetName());


    // we want to get rid of the labels in the node selector widgets.
    // instead of using these, we're using the menubuttons to
    // dual as labels and as containers for configuration options for each layer.
    this->Script ( "pack forget %s", this->BackgroundSelector->GetLabel()->GetWidgetName() );
    this->Script ( "pack forget %s", this->ForegroundSelector->GetLabel()->GetWidgetName() );
    this->Script ( "pack forget %s", this->LabelSelector->GetLabel()->GetWidgetName() );
    this->Script ( "pack forget %s", this->OrientationSelector->GetLabel()->GetWidgetName() );

    // and put observers on widgets
    this->AddWidgetObservers();
}


//----------------------------------------------------------------------------
void vtkSlicerSliceControllerWidget::UpdateLayerMenus()
{
  // ensure that the interpolation checkbutton state matches the node
  int interp;
  if ( this->SliceLogic )
    {
    if ( this->SliceLogic->GetBackgroundLayer() )
      {
      if ( this->SliceLogic->GetBackgroundLayer()->GetVolumeDisplayNode() )
        {
        interp = this->SliceLogic->GetBackgroundLayer()->GetVolumeDisplayNode()->GetInterpolate();
        if ( interp != this->BackgroundMenuButton->GetMenu()->GetItemSelectedState ( "interpolation" ) )
          {
          this->BackgroundMenuButton->GetMenu()->SetItemSelectedState ( "interpolation",
                                                                        this->SliceLogic->GetBackgroundLayer()->GetVolumeDisplayNode()->GetInterpolate() );
          }
        }
      }
    if ( this->SliceLogic->GetForegroundLayer() )
      {
      if ( this->SliceLogic->GetForegroundLayer()->GetVolumeDisplayNode() )
        {
        interp = this->SliceLogic->GetForegroundLayer()->GetVolumeDisplayNode()->GetInterpolate();
        if ( interp != this->ForegroundMenuButton->GetMenu()->GetItemSelectedState ( "interpolation" ) )
          {
          this->ForegroundMenuButton->GetMenu()->SetItemSelectedState ( "interpolation",
                                                                        this->SliceLogic->GetForegroundLayer()->GetVolumeDisplayNode()->GetInterpolate() );
          }
        }
      }
    }
}



//----------------------------------------------------------------------------
void vtkSlicerSliceControllerWidget::UpdateOrientation ( int link )
{
  int i, nnodes;
  vtkMRMLSliceNode *snode;
  
  //--- if slice viewers are linked, modify all Controller's SliceNodes;
  //--- otherwise just update the one we know about.
  vtkKWMenuButton *mb = this->OrientationSelector->GetWidget()->GetWidget();
  if ( !strcmp (mb->GetValue(), "Axial") )   
    {
    if ( link )
      {
      nnodes = this->GetMRMLScene()->GetNumberOfNodesByClass ( "vtkMRMLSliceNode");
      for ( i=0; i<nnodes; i++)
        {
        snode = vtkMRMLSliceNode::SafeDownCast (
                                                this->GetMRMLScene()->GetNthNodeByClass (i, "vtkMRMLSliceNode"));
        snode->SetOrientationToAxial();
        }
      }
    else
      {
      this->SliceNode->SetOrientationToAxial();
      }
    }
  if ( !strcmp (mb->GetValue(), "Sagittal") )   
    {
    if ( link )
      {
      nnodes = this->GetMRMLScene()->GetNumberOfNodesByClass ( "vtkMRMLSliceNode");          
      for ( i=0; i<nnodes; i++)
        {
        snode = vtkMRMLSliceNode::SafeDownCast (
                                                this->GetMRMLScene()->GetNthNodeByClass (i, "vtkMRMLSliceNode"));
        snode->SetOrientationToSagittal();
        }
      }
    else
      {
      this->SliceNode->SetOrientationToSagittal();
      }
    }
  if ( !strcmp (mb->GetValue(), "Coronal") )   
    {
    if ( link )
      {
      nnodes = this->GetMRMLScene()->GetNumberOfNodesByClass ( "vtkMRMLSliceNode");          
      for ( i=0; i<nnodes; i++)
        {
        snode = vtkMRMLSliceNode::SafeDownCast (
                                                this->GetMRMLScene()->GetNthNodeByClass (i, "vtkMRMLSliceNode"));
        snode->SetOrientationToCoronal();
        }
      }
    else
      {
      this->SliceNode->SetOrientationToCoronal();
      }
    }

}


//----------------------------------------------------------------------------
void vtkSlicerSliceControllerWidget::UpdateForegroundLayer ( int link )
{
  int i, nnodes;
  vtkMRMLSliceCompositeNode *cnode;
  
  //--- if slice viewers are linked, modify all Controller's SliceCompositeNodes
  //--- otherwise, just update the one we know about.
  if ( link )
    {
    nnodes = this->GetMRMLScene()->GetNumberOfNodesByClass ( "vtkMRMLSliceCompositeNode");          
    for ( i=0; i<nnodes; i++)
      {
      cnode = vtkMRMLSliceCompositeNode::SafeDownCast (
                                                       this->GetMRMLScene()->GetNthNodeByClass (i, "vtkMRMLSliceCompositeNode"));
      if ( cnode != NULL && this->ForegroundSelector->GetSelected() != NULL )
        {
        cnode->SetForegroundVolumeID( this->ForegroundSelector->GetSelected()->GetID() );
        }
      else if ( cnode != NULL )
        {
        cnode->SetForegroundVolumeID ( NULL );
        }
      }
    }
  else
    {
    if  (this->ForegroundSelector->GetSelected() != NULL && this->SliceCompositeNode != NULL)
      {
      this->SliceCompositeNode->SetForegroundVolumeID( this->ForegroundSelector->GetSelected()->GetID() );
      } 
    else if (this->SliceCompositeNode != NULL)
      {
      this->SliceCompositeNode->SetForegroundVolumeID( NULL );
      }
    }

}

//----------------------------------------------------------------------------
void vtkSlicerSliceControllerWidget::UpdateBackgroundLayer ( int link )
{
  int i, nnodes;
  vtkMRMLSliceCompositeNode *cnode;
  
  //--- if slice viewers are linked, modify all Controller's SliceCompositeNodes
  //--- otherwise, just update the one we know about.
  if ( link )
    {
    nnodes = this->GetMRMLScene()->GetNumberOfNodesByClass ( "vtkMRMLSliceCompositeNode");          
    for ( i=0; i<nnodes; i++)
      {
      cnode = vtkMRMLSliceCompositeNode::SafeDownCast (
                                                       this->GetMRMLScene()->GetNthNodeByClass (i, "vtkMRMLSliceCompositeNode"));
      if ( cnode != NULL && this->BackgroundSelector->GetSelected() != NULL)
        {
        cnode->SetBackgroundVolumeID( this->BackgroundSelector->GetSelected()->GetID() );
        }
      else if ( cnode != NULL )
        {
        cnode->SetBackgroundVolumeID ( NULL );
        }
      }
    }
  else
    {
    if  (this->BackgroundSelector->GetSelected() != NULL && this->SliceCompositeNode != NULL)
      {
      this->SliceCompositeNode->SetBackgroundVolumeID( this->BackgroundSelector->GetSelected()->GetID() );
      } 
    else if (this->SliceCompositeNode != NULL)
      {
      this->SliceCompositeNode->SetBackgroundVolumeID( NULL );
      }
    }

}

//----------------------------------------------------------------------------
void vtkSlicerSliceControllerWidget::UpdateLabelLayer ( int link )
{

  int i, nnodes;
  vtkMRMLSliceCompositeNode *cnode;
  
  //--- if slice viewers are linked, modify all Controller's SliceCompositeNodes
  //--- otherwise, just update the one we know about.
  if ( link )
    {
    nnodes = this->GetMRMLScene()->GetNumberOfNodesByClass ( "vtkMRMLSliceCompositeNode");          
    for ( i=0; i<nnodes; i++)
      {
      cnode = vtkMRMLSliceCompositeNode::SafeDownCast (
                                                       this->GetMRMLScene()->GetNthNodeByClass (i, "vtkMRMLSliceCompositeNode"));
      if ( cnode != NULL && this->LabelSelector->GetSelected() != NULL )
        {
        cnode->SetLabelVolumeID( this->LabelSelector->GetSelected()->GetID() );
        }
      else if ( cnode != NULL )
        {
        cnode->SetLabelVolumeID ( NULL );
        }
      }
    }
  else
    {
    if  (this->LabelSelector->GetSelected() != NULL && this->SliceCompositeNode != NULL)
      {
      this->SliceCompositeNode->SetLabelVolumeID( this->LabelSelector->GetSelected()->GetID() );
      } 
    else if (this->SliceCompositeNode != NULL)
      {
      this->SliceCompositeNode->SetLabelVolumeID( NULL );
      }
    }
}


//----------------------------------------------------------------------------
void vtkSlicerSliceControllerWidget::RaiseVolumeDisplayPanel ( char *id )
{
  vtkSlicerVolumesGUI *vgui;
  vtkSlicerApplication *app;
  vtkSlicerApplicationGUI *appgui;
    
  app = vtkSlicerApplication::SafeDownCast (this->GetApplication());
  vgui = vtkSlicerVolumesGUI::SafeDownCast ( app->GetModuleGUIByName ("Volumes") );
  vgui->GetApplicationLogic()->GetSelectionNode()->SetActiveVolumeID ( id );
  vgui->GetVolumeDisplayWidget()->SetVolumeNode (vtkMRMLVolumeNode::SafeDownCast (this->GetMRMLScene()->GetNodeByID ( id )) );
  appgui = vgui->GetApplicationGUI ( );
  appgui->SelectModule ( "Volumes" );
  //vgui->GetHelpAndAboutFrame()->CollapseFrame();
//  vgui->GetLoadFrame()->CollapseFrame();
  vgui->GetDisplayFrame()->ExpandFrame();
//  vgui->GetSaveFrame()->CollapseFrame();
}




//----------------------------------------------------------------------------
void vtkSlicerSliceControllerWidget::FitSliceToBackground ( int link )
{

  vtkSlicerSlicesGUI *ssgui;
  vtkSlicerSliceGUI *sgui;
  vtkSlicerApplication *app;
  vtkSlicerApplicationGUI *appGUI;
  int found = 0;
    
  // find the sliceGUI for this controller
  app = vtkSlicerApplication::SafeDownCast (this->GetApplication());
  ssgui = vtkSlicerSlicesGUI::SafeDownCast ( app->GetModuleGUIByName ("Slices") );
  appGUI = ssgui->GetApplicationGUI ( );

  if ( ssgui->GetSliceGUICollection() )
    {
    ssgui->GetSliceGUICollection()->InitTraversal();
    sgui = vtkSlicerSliceGUI::SafeDownCast ( ssgui->GetSliceGUICollection()->GetNextItemAsObject() );
    while ( sgui != NULL )
      {
      if (sgui->GetSliceController() == this )
        {
        found = 1;
        break;
        }
      sgui = vtkSlicerSliceGUI::SafeDownCast ( ssgui->GetSliceGUICollection()->GetNextItemAsObject() );
      }
    }
  else
    {
    return;
    }

  if ( found )
    {
    if ( link )
      {
      // First save all SliceNodes for undo:
      ssgui->GetSliceGUICollection()->InitTraversal();
      sgui = vtkSlicerSliceGUI::SafeDownCast ( ssgui->GetSliceGUICollection()->GetNextItemAsObject() );
      vtkCollection *nodes = vtkCollection::New();
      while ( sgui != NULL )
        {
        nodes->AddItem ( sgui->GetSliceNode ( ) );
        sgui = vtkSlicerSliceGUI::SafeDownCast ( ssgui->GetSliceGUICollection()->GetNextItemAsObject() );
        }
      this->MRMLScene->SaveStateForUndo ( nodes );
      nodes->Delete ( );

      // Now fit all Slices to background
      ssgui->GetSliceGUICollection()->InitTraversal();
      sgui = vtkSlicerSliceGUI::SafeDownCast ( ssgui->GetSliceGUICollection()->GetNextItemAsObject() );
      while ( sgui != NULL )
        {
        int w, h;
        //w = sgui->GetSliceViewer()->GetRenderWidget ( )->GetWidth();
        //h = sgui->GetSliceViewer()->GetRenderWidget ( )->GetHeight();
        sscanf(
          this->Script("winfo width %s", 
              sgui->GetSliceViewer()->GetRenderWidget ( )->GetWidgetName()), 
          "%d", &w);
        sscanf(
          this->Script("winfo height %s", 
              sgui->GetSliceViewer()->GetRenderWidget ( )->GetWidgetName()), 
          "%d", &h);
        sgui->GetLogic()->FitSliceToBackground ( w, h );
        sgui->GetSliceNode()->UpdateMatrices( );
        appGUI->GetSlicesControlGUI()->RequestFOVEntriesUpdate();
        sgui = vtkSlicerSliceGUI::SafeDownCast ( ssgui->GetSliceGUICollection()->GetNextItemAsObject() );
        }
      }
    else
      {
      this->MRMLScene->SaveStateForUndo ( this->SliceNode );

      int w, h;
      // gives bogus values:
      //w = sgui->GetSliceViewer()->GetRenderWidget ( )->GetWidth();
      //h = sgui->GetSliceViewer()->GetRenderWidget ( )->GetHeight();

      sscanf(
        this->Script("winfo width %s", 
            sgui->GetSliceViewer()->GetRenderWidget ( )->GetWidgetName()), 
        "%d", &w);
      sscanf(
        this->Script("winfo height %s", 
            sgui->GetSliceViewer()->GetRenderWidget ( )->GetWidgetName()), 
        "%d", &h);

      sgui->GetLogic()->FitSliceToBackground ( w, h );
      this->SliceNode->UpdateMatrices( );
      appGUI->GetSlicesControlGUI()->RequestFOVEntriesUpdate();      
      }
    }
}



//----------------------------------------------------------------------------
void vtkSlicerSliceControllerWidget::ProcessWidgetEvents ( vtkObject *caller, unsigned long event, void *callData ) 
{ 
  //
  // Is this Slice controller linked to others?
  //
  int link, i, nnodes;
  vtkMRMLSliceCompositeNode *cnode;
  vtkMRMLSliceNode *snode;
  int numHPanes, numVPanes;
    
  //
  // --- Find out whether SliceViewers are linked or unlinked
  // --- so we know how to handle control.
  //
  if ( this->SliceCompositeNode )
    {
    link = this->SliceCompositeNode->GetLinkedControl ( );
    }
  else
    {
    link = 0;
    }
  
  //
  // --- Get a route to all SliceGUI's SliceNodes and SliceCompositeNodes in case of link
  //
  vtkSlicerApplication *app = vtkSlicerApplication::SafeDownCast (this->GetApplication());
  vtkSlicerSlicesGUI *sgui;
  if (app)
    {
    sgui = vtkSlicerSlicesGUI::SafeDownCast ( app->GetModuleGUIByName("Slices"));
    }
  else
    {
    sgui = NULL;
    }


  //
  // --- Set the slice node to null if it no longer exists in the MRML scene.
  //
   if (this->SliceNode != NULL &&
       this->MRMLScene->GetNodeByID(this->SliceNode->GetID()) == NULL)
    {
    this->SetSliceNode(NULL);
    }
   //
   // --- Set the slice composite node to null if it no longer exists in the scene.
   //
  if (this->SliceCompositeNode != NULL &&
      this->MRMLScene->GetNodeByID(this->SliceCompositeNode->GetID()) == NULL)
    {
    this->SetSliceCompositeNode(NULL);
    }

  //
  // --- Update orientation if needed
  //
  if ( this->SliceNode != NULL && 
       vtkKWMenu::SafeDownCast(caller) == this->OrientationSelector->GetWidget()->GetWidget()->GetMenu() )
    {
    if (sgui)
      this->UpdateOrientation ( link );
    }

  //
  // Update Foreground if needed
  //
  if ( vtkSlicerNodeSelectorWidget::SafeDownCast(caller) == this->ForegroundSelector )
    {
    if (sgui )
      {
      this->UpdateForegroundLayer ( link );
      this->ForegroundSelector->SetBalloonHelpString ("Select the foreground");
      }
    }

  //
  // Update Background if needed
  //
  if ( vtkSlicerNodeSelectorWidget::SafeDownCast(caller) == this->BackgroundSelector )
    {
    if (sgui )
      {
      this->UpdateBackgroundLayer ( link );
      this->BackgroundSelector->SetBalloonHelpString ("Select the background");
      }
    }

  
  //
  // Update Label layer if needed
  //
  if ( vtkSlicerNodeSelectorWidget::SafeDownCast(caller) == this->LabelSelector )
    {
    if (sgui)
      {
      this->UpdateLabelLayer ( link );
      this->LabelSelector->SetBalloonHelpString ("Select the label map");
      }
    }
  
  if ( !this->SliceNode)
    {
    return;
    }
  
  //
  // Was event invoked by other widgets? (button, scale, entry, menu)?
  //
  vtkKWPushButton *button = vtkKWPushButton::SafeDownCast ( caller );
  vtkKWScale *scale = vtkKWScale::SafeDownCast (caller);
  vtkKWEntry *entry = vtkKWEntry::SafeDownCast(caller);
  vtkKWMenu *menu = vtkKWMenu::SafeDownCast(caller);

  // Toggle the SliceNode's visibility.
  if ( button == this->GetVisibilityToggle() && event == vtkKWPushButton::InvokedEvent )
    {
    //--- if slice viewers are linked, modify all Controller's SliceNodes
    int vis = this->SliceNode->GetSliceVisible();
    if ( link )
      {
      nnodes = this->GetMRMLScene()->GetNumberOfNodesByClass ( "vtkMRMLSliceNode");          
      for ( i=0; i<nnodes; i++)
        {
        snode = vtkMRMLSliceNode::SafeDownCast (
                                                this->GetMRMLScene()->GetNthNodeByClass (i, "vtkMRMLSliceNode"));
        this->MRMLScene->SaveStateForUndo ( snode );
        snode->SetSliceVisible ( !vis );
        }
      }
    else
      {
      this->MRMLScene->SaveStateForUndo ( this->SliceNode );
      this->SliceNode->SetSliceVisible ( !vis );
      }
    }

  //
  // Toggle the Linked control or pop up opacity scale.
  //
  if ( button == this->GetLinkButton() && event == vtkKWPushButton::InvokedEvent )
    {
    this->MRMLScene->SaveStateForUndo ( this->SliceNode );
    this->ToggleSlicesLink ( );
    }
  else if ( button == this->LabelOpacityButton && event == vtkKWPushButton::InvokedEvent )
    {
    this->PopUpLabelOpacityScaleAndEntry();
    }
  else if ( button == this->FitToWindowButton && event == vtkKWPushButton::InvokedEvent )
    {
     this->FitSliceToBackground ( link );
    }
  
  //
  // Raise volumes module if adjustment to Volume display is requested.
  //
  if ( menu == this->VolumeDisplayMenuButton->GetMenu() &&
       event == vtkKWMenu::MenuItemInvokedEvent && app )
    {
    char *id = NULL;
    if ( !strcmp (this->VolumeDisplayMenuButton->GetValue(), "Foreground volume") )
      {
      this->MRMLScene->SaveStateForUndo ( this->SliceNode );
      // raise volumes module with foreground
      id = this->SliceCompositeNode->GetForegroundVolumeID( );
      if ( id )
        {
        this->RaiseVolumeDisplayPanel ( id );
        }
      }
    else if (!strcmp (this->VolumeDisplayMenuButton->GetValue(), "Background volume"))
      {
      this->MRMLScene->SaveStateForUndo ( this->SliceNode );
      // raise volumes module with background
      id = this->SliceCompositeNode->GetBackgroundVolumeID( );
      if ( id )
        {
        this->RaiseVolumeDisplayPanel ( id );
        }
      }
    else if (!strcmp(this->VolumeDisplayMenuButton->GetValue(), "Label map"))
      {
      this->MRMLScene->SaveStateForUndo ( this->SliceNode );
      // raise volumes module with label
      id = this->SliceCompositeNode->GetLabelVolumeID( );
      if ( id )
        {
        this->RaiseVolumeDisplayPanel ( id );
        }
      }
    }
  else if ( menu == this->ForegroundMenuButton->GetMenu() &&
            event == vtkKWMenu::MenuItemInvokedEvent && app )
    {
    //
    // check to see if this layer's associated VolumeDisplayNode has
    // a different value for interpolation than checkbutton indicates.
    //
    int interp = this->ForegroundMenuButton->GetMenu()->GetItemSelectedState ( "interpolation" );
    if ( this->SliceLogic )
      {
      if ( this->SliceLogic->GetForegroundLayer() )
        {
        if ( this->SliceLogic->GetForegroundLayer()->GetVolumeDisplayNode() )
          {
          if ( interp != this->SliceLogic->GetForegroundLayer()->GetVolumeDisplayNode()->GetInterpolate() )
            {
            // save state for undo and modify the node's value to match GUI
            this->MRMLScene->SaveStateForUndo ( this->SliceLogic->GetForegroundLayer()->GetVolumeDisplayNode() );
            this->SliceLogic->GetForegroundLayer()->GetVolumeDisplayNode()->SetInterpolate ( interp );
            // need this to propagate change thru to the VolumeDisplayWidget's GUI
            if ( this->SliceLogic->GetForegroundLayer()->GetVolumeNode() )
              {
              this->SliceLogic->GetForegroundLayer()->GetVolumeNode()->Modified();
              }
            }
          }
        }
      }
      // change other slice controller widgets if linked; go thru the SlicesGUI collection
    if ( link )
      {
      vtkSlicerSlicesGUI *ssgui = vtkSlicerSlicesGUI::SafeDownCast ( app->GetModuleGUIByName ("Slices") );
      vtkSlicerSliceGUI *sgui;
      if ( ssgui->GetSliceGUICollection() )
        {
        ssgui->GetSliceGUICollection()->InitTraversal();
        sgui = vtkSlicerSliceGUI::SafeDownCast ( ssgui->GetSliceGUICollection()->GetNextItemAsObject() );
        while ( sgui ) 
          {
          if ( sgui->GetLogic() )
            {
            if  ( sgui->GetLogic()->GetForegroundLayer() )
              {
              if ( sgui->GetLogic()->GetForegroundLayer()->GetVolumeDisplayNode() )
                {
                if ( interp != sgui->GetLogic()->GetForegroundLayer()->GetVolumeDisplayNode()->GetInterpolate() )
                  {
                  this->MRMLScene->SaveStateForUndo ( sgui->GetLogic()->GetForegroundLayer()->GetVolumeDisplayNode() );
                  sgui->GetLogic()->GetForegroundLayer()->GetVolumeDisplayNode()->SetInterpolate ( interp );
                  // need this to propagate change thru to the VolumeDisplayWidget's GUI
                  if ( sgui->GetLogic()->GetForegroundLayer()->GetVolumeNode() )
                    {
                    sgui->GetLogic()->GetForegroundLayer()->GetVolumeNode()->Modified();
                    }
                  }
                }
              }
            }
          sgui = vtkSlicerSliceGUI::SafeDownCast ( ssgui->GetSliceGUICollection()->GetNextItemAsObject() );
          }
        }
      }
    }  

  else if ( menu == this->BackgroundMenuButton->GetMenu() &&
            event == vtkKWMenu::MenuItemInvokedEvent && app )
    {
    // check to see if this layer's associated VolumeDisplayNode has
    // a different value for interpolation than checkbutton indicates.
    //
    int interp = this->BackgroundMenuButton->GetMenu()->GetItemSelectedState ( "interpolation" );
    if ( this->SliceLogic )
      {
      if ( this->SliceLogic->GetBackgroundLayer() )
        {
        if ( this->SliceLogic->GetBackgroundLayer()->GetVolumeDisplayNode() )
          {
          if ( interp != this->SliceLogic->GetBackgroundLayer()->GetVolumeDisplayNode()->GetInterpolate() )
            {
            // save state for undo and modify the node's value to match GUI
            this->MRMLScene->SaveStateForUndo ( this->SliceLogic->GetBackgroundLayer()->GetVolumeDisplayNode() );
            this->SliceLogic->GetBackgroundLayer()->GetVolumeDisplayNode()->SetInterpolate ( interp );
            // need this to propagate change thru to the VolumeDisplayWidget's GUI
            if ( this->SliceLogic->GetBackgroundLayer()->GetVolumeNode() )
              {
              this->SliceLogic->GetBackgroundLayer()->GetVolumeNode()->Modified();
              }
            }
          }
        }
      }
      // change other slice controller widgets if linked; go thru the SlicesGUI collection
    if ( link )
      {
      vtkSlicerSlicesGUI *ssgui = vtkSlicerSlicesGUI::SafeDownCast ( app->GetModuleGUIByName ("Slices") );
      vtkSlicerSliceGUI *sgui;
      if ( ssgui->GetSliceGUICollection() )
        {
        ssgui->GetSliceGUICollection()->InitTraversal();
        sgui = vtkSlicerSliceGUI::SafeDownCast ( ssgui->GetSliceGUICollection()->GetNextItemAsObject() );
        while ( sgui ) 
          {
          if ( sgui->GetLogic() )
            {
            if  ( sgui->GetLogic()->GetBackgroundLayer() )
              {
              if ( sgui->GetLogic()->GetBackgroundLayer()->GetVolumeDisplayNode() )
                {
                if ( interp != sgui->GetLogic()->GetBackgroundLayer()->GetVolumeDisplayNode()->GetInterpolate() )
                  {
                  this->MRMLScene->SaveStateForUndo ( sgui->GetLogic()->GetBackgroundLayer()->GetVolumeDisplayNode() );
                  sgui->GetLogic()->GetBackgroundLayer()->GetVolumeDisplayNode()->SetInterpolate ( interp );
                  // need this to propagate change thru to the VolumeDisplayWidget's GUI
                  if ( sgui->GetLogic()->GetBackgroundLayer()->GetVolumeNode() )
                    {
                    sgui->GetLogic()->GetBackgroundLayer()->GetVolumeNode()->Modified();
                    }
                  }
                }
              }
            }
          sgui = vtkSlicerSliceGUI::SafeDownCast ( ssgui->GetSliceGUICollection()->GetNextItemAsObject() );
          }
        }
      }
    }
  else if ( menu == this->LightboxButton->GetMenu() &&
            event == vtkKWMenu::MenuItemInvokedEvent && app )
    {
    //
    // Lightbox view functionality developing here.
    //
    if ( !strcmp ( this->LightboxButton->GetValue(), "1x1 view") )
      {
      if ( link && sgui )
        {
        // apply this reformat to all slice MRML
        }
      else
        {
        // apply this reformat to only this slice MRML
        this->SliceNode->SetLayoutGrid( 1, 1 );
        }
      }
    else if ( !strcmp ( this->LightboxButton->GetValue(), "2x2 view") )
      {
      if ( link && sgui )
        {
        // apply this reformat to all slice MRMLs
        }
      else
        {
        // apply this reformat to only this slice MRML
        this->SliceNode->SetLayoutGrid( 2, 2 );
        }
      }
    else if  ( !strcmp ( this->LightboxButton->GetValue(), "3x3 view" ) )
      {
      if ( link && sgui )
        {
        // apply this reformat to all slice MRMLs
        }
      else
        {
        // apply this reformat to only this slice MRML
        this->SliceNode->SetLayoutGrid( 3, 3 );
        }
      }
    else if ( !strcmp ( this->LightboxButton->GetValue (), "6x6 view") )
      {
      if ( link && sgui )
        {
        // apply this reformat to all slice MRMLs
        }
      else
        {
        // apply this reformat to only this slice MRML
        this->SliceNode->SetLayoutGrid( 6, 6 );
        }
      }
    else if ( !strcmp ( this->LightboxButton->GetValue (), "customized view") )
      {
      // pop up a toplevel to specify NXM view
      PopUpLightboxCustomLayoutFrame();
      }
    }
  if ( button == this->LightboxApplyButton &&
            event == vtkKWPushButton::InvokedEvent )
    {
    numHPanes = this->LightboxWidthEntry->GetValueAsInt();
    numVPanes = this->LightboxHeightEntry->GetValueAsInt();
    if ( link && sgui )
      {
      // apply this reformat to all slice MRMLs
      }
    else
      {
      // apply this reformat to only this slice MRML
      this->SliceNode->SetLayoutGrid( numHPanes, numVPanes );
      }
    }
  
  //
  // Scales starting to move? save state for undo.
  //
  if ( this->OffsetScale->GetWidget() == vtkKWScale::SafeDownCast( caller ) &&
          event == vtkKWScale::ScaleValueStartChangingEvent )
    {
    // set an undo state when the scale starts being dragged
    this->MRMLScene->SaveStateForUndo( this->SliceNode );
    }
  else if ( scale == this->LabelOpacityScale->GetWidget() && event == vtkKWScale::ScaleValueStartChangingEvent )
    {
    if ( link && sgui ) 
      {
      nnodes = this->GetMRMLScene()->GetNumberOfNodesByClass ( "vtkMRMLSliceCompositeNode");          
      for ( i=0; i<nnodes; i++)
        {
        // set an undo state when the scale starts being dragged
        cnode = vtkMRMLSliceCompositeNode::SafeDownCast (
                                                this->GetMRMLScene()->GetNthNodeByClass (i, "vtkMRMLSliceCompositeNode"));
        this->MRMLScene->SaveStateForUndo( cnode );
        }
      }
    else
      {
      this->MRMLScene->SaveStateForUndo ( this->SliceCompositeNode );
      }
    }

  //
  // Scales are moving? modify MRML
  //
  int modified = 0;
  if ( (double) this->LabelOpacityScale->GetValue() != this->SliceCompositeNode->GetLabelOpacity() )
    {
    //--- if slice viewers are linked, modify all Controller's SliceCompositeNodes.
    if ( link && sgui )
      {
      nnodes = this->GetMRMLScene()->GetNumberOfNodesByClass ( "vtkMRMLSliceCompositeNode");          
      for ( i=0; i<nnodes; i++)
        {
        cnode = vtkMRMLSliceCompositeNode::SafeDownCast (
                                                this->GetMRMLScene()->GetNthNodeByClass (i, "vtkMRMLSliceCompositeNode"));
        cnode->SetLabelOpacity ( (double) this->LabelOpacityScale->GetValue() );
        }
      }
    else
      {
      this->SliceCompositeNode->SetLabelOpacity ( (double) this->LabelOpacityScale->GetValue() );
//      modified = 1;
      }
    }
  if ( (double) this->OffsetScale->GetValue() != this->SliceLogic->GetSliceOffset() )
    {
    this->SliceLogic->SetSliceOffset( (double) this->OffsetScale->GetValue() );
    modified = 1;
    }

  if ( modified )
    {
    this->Modified();
    }

}



//---------------------------------------------------------------------------
void vtkSlicerSliceControllerWidget::HideLightboxCustomLayoutFrame ( )
{
  if ( !this->LightboxTopLevel )
    {
    return;
    }
  this->LightboxTopLevel->Withdraw();

}

//---------------------------------------------------------------------------
void vtkSlicerSliceControllerWidget::PopUpLightboxCustomLayoutFrame()
{
  if ( !this->LightboxButton || !this->LightboxButton->IsCreated())
    {
    return;
    }

  // Get the position of the mouse, the position and size of the push button,
  // the size of the scale.

  int x, y, px, py, ph;
  vtkSlicerApplication *app = vtkSlicerApplication::SafeDownCast ( this->GetApplication());
  
  vtkKWTkUtilities::GetMousePointerCoordinates(this->LightboxButton, &x, &y);
  vtkKWTkUtilities::GetWidgetCoordinates(this->LabelOpacityButton, &px, &py);
  vtkKWTkUtilities::GetWidgetSize(this->LightboxButton, NULL, &ph);
 
  this->LightboxTopLevel->SetPosition(px-ph, py+ph);
  app->ProcessPendingEvents();
  this->LightboxTopLevel->DeIconify();
  this->LightboxTopLevel->Raise();

}


//---------------------------------------------------------------------------
void vtkSlicerSliceControllerWidget::HideLabelOpacityScaleAndEntry ( )
{
  if ( !this->LabelOpacityTopLevel )
    {
    return;
    }
  this->LabelOpacityTopLevel->Withdraw();
}


//---------------------------------------------------------------------------
void vtkSlicerSliceControllerWidget::PopUpLabelOpacityScaleAndEntry ( )
{
  if ( !this->LabelOpacityButton || !this->LabelOpacityButton->IsCreated())
    {
    return;
    }

  // Get the position of the mouse, the position and size of the push button,
  // the size of the scale.

  int x, y, py, ph, scx, scy, sx, sy;
  vtkSlicerApplication *app = vtkSlicerApplication::SafeDownCast ( this->GetApplication());
  
  vtkKWTkUtilities::GetMousePointerCoordinates(this->LabelOpacityButton, &x, &y);
  vtkKWTkUtilities::GetWidgetCoordinates(this->LabelOpacityButton, NULL, &py);
  vtkKWTkUtilities::GetWidgetSize(this->LabelOpacityButton, NULL, &ph);
  vtkKWTkUtilities::GetWidgetRelativeCoordinates(this->LabelOpacityScale->GetScale(), &sx, &sy);
  sscanf(this->Script("%s coords %g", this->LabelOpacityScale->GetScale()->GetWidgetName(),
                      this->LabelOpacityScale->GetScale()->GetValue()), "%d %d", &scx, &scy);
 
  // Place the scale so that the slider is coincident with the x mouse position
  // and just below the push button
  x -= sx + scx;
  if (py <= y && y <= (py + ph -1))
    {
    y = py + ph - 3;
    }
  else
    {
    y -= sy + scy;
    }

  this->LabelOpacityTopLevel->SetPosition(x, y);
  app->ProcessPendingEvents();
  this->LabelOpacityTopLevel->DeIconify();
  this->LabelOpacityTopLevel->Raise();
}

//----------------------------------------------------------------------------
void vtkSlicerSliceControllerWidget::LinkAllSlices  ( )
{
  vtkSlicerApplication *app = vtkSlicerApplication::SafeDownCast (this->GetApplication());
  vtkSlicerSlicesGUI *sgui = vtkSlicerSlicesGUI::SafeDownCast ( app->GetModuleGUIByName("Slices"));
  vtkMRMLSliceCompositeNode *cnode;
  if ( app && sgui )
    {
    // link all slice controllers
    int nnodes = this->GetMRMLScene()->GetNumberOfNodesByClass ( "vtkMRMLSliceCompositeNode");
    for ( int i=0; i<nnodes; i++)
      {
      cnode = vtkMRMLSliceCompositeNode::SafeDownCast (
                                                this->GetMRMLScene()->GetNthNodeByClass (i, "vtkMRMLSliceCompositeNode"));
      cnode->SetLinkedControl ( 1 );
      }
    }
}

//----------------------------------------------------------------------------
void vtkSlicerSliceControllerWidget::UnlinkAllSlices  ( )
{
  vtkSlicerApplication *app = vtkSlicerApplication::SafeDownCast (this->GetApplication());
  vtkSlicerSlicesGUI *sgui = vtkSlicerSlicesGUI::SafeDownCast ( app->GetModuleGUIByName("Slices"));
  vtkMRMLSliceCompositeNode *cnode;
  if ( app && sgui )
    {
    // unlink all slice controllers.
    int nnodes = this->GetMRMLScene()->GetNumberOfNodesByClass ( "vtkMRMLSliceCompositeNode");
    for ( int i=0; i<nnodes; i++)
      {
      cnode = vtkMRMLSliceCompositeNode::SafeDownCast (
                                                this->GetMRMLScene()->GetNthNodeByClass (i, "vtkMRMLSliceCompositeNode"));
      cnode->SetLinkedControl ( 0 );
      }
    }
}

//----------------------------------------------------------------------------
int vtkSlicerSliceControllerWidget::AllSlicesLinked ( )
{
  vtkSlicerApplication *app = vtkSlicerApplication::SafeDownCast (this->GetApplication());
  vtkSlicerSlicesGUI *sgui = vtkSlicerSlicesGUI::SafeDownCast ( app->GetModuleGUIByName("Slices"));
  vtkMRMLSliceCompositeNode *cnode;
  int link = 1;
  if ( app && sgui )
    {
    // are all slice controllers linked? assume they are
    // unless we find one that's unliked.
    int nnodes = this->GetMRMLScene()->GetNumberOfNodesByClass ( "vtkMRMLSliceCompositeNode");
    for ( int i=0; i<nnodes; i++)
      {
      cnode = vtkMRMLSliceCompositeNode::SafeDownCast (
                                                       this->GetMRMLScene()->GetNthNodeByClass (i, "vtkMRMLSliceCompositeNode"));
      if ( cnode->GetLinkedControl ( ) == 0 )
        {
        link = 0;
        }
      }
    }
  return ( link );
}


//----------------------------------------------------------------------------
void vtkSlicerSliceControllerWidget::ToggleSlicesLink  ( )
{
  // check to see what the current link status is.
  int link = this->SliceCompositeNode->GetLinkedControl();

  if ( link==1 )
    {
    this->UnlinkAllSlices();
    }
  else
    {
    // slices are currently unlinked, so link them all.
    this->LinkAllSlices();
    }
}




//----------------------------------------------------------------------------
void vtkSlicerSliceControllerWidget::ProcessMRMLEvents ( vtkObject *caller, unsigned long event, void *callData ) 
{ 
  if (this->SliceNode != NULL && this->MRMLScene->GetNodeByID(this->SliceNode->GetID()) == NULL)
    {
    this->SetSliceNode(NULL);
    }

  if (this->SliceCompositeNode != NULL && this->MRMLScene->GetNodeByID(this->SliceCompositeNode->GetID()) == NULL)
    {
    this->SetSliceCompositeNode(NULL);
    }

  if ( !this->SliceNode)
    {
    return;
    }
   if ( !this->SliceCompositeNode)
     {
     return;
     }
  
  int modified = 0;


  // 
  // Update the menu to match the node
  //
  vtkKWMenuButton *mb = this->OrientationSelector->GetWidget()->GetWidget();
  mb->SetValue( this->SliceNode->GetOrientationString() );

  //
  // Make the Balloon help hint about which way scrolling works.
  //
  if ( !(strcmp(this->SliceNode->GetOrientationString(), "Axial")))
    {
    // Orientation is Axial: I <----> S
    this->OffsetScale->GetScale()->SetBalloonHelpString ( "I <-----> S" );
    }
  else if ( !(strcmp(this->SliceNode->GetOrientationString(), "Sagittal")))
    {
    // Orientation is Sagittal: R <----> L  
    this->OffsetScale->GetScale()->SetBalloonHelpString ( "R <-----> L" );
    }
  else if ( !(strcmp(this->SliceNode->GetOrientationString(), "Coronal")))
    {
    // Orientation is Coronal: P <----> A
    this->OffsetScale->GetScale()->SetBalloonHelpString ( "P <-----> A" );
    }
  else
    {
    // Orientation is Oblique: make tooltip null
    this->OffsetScale->GetScale()->SetBalloonHelpString ( "" ) ;
    }

  //
  // Set the scale increments to match the z spacing (rotated
  // into slice space)
  //
  const double *sliceSpacing;
  sliceSpacing = this->SliceLogic->GetBackgroundSliceSpacing();

  this->OffsetScale->SetResolution(sliceSpacing[2]);
  this->Script ("%s configure -digits 20", 
                this->OffsetScale->GetScale()->GetWidgetName());


  //
  // Set the scale range to match the field of view
  //
  double sliceBounds[6];
  this->SliceLogic->GetBackgroundSliceBounds(sliceBounds);

  double fovover2 = this->SliceNode->GetFieldOfView()[2] / 2.;
  double newMin = sliceBounds[4];
  double newMax = sliceBounds[5];
  double min, max;
  this->OffsetScale->GetRange(min, max);
  if ( min != newMin || max != newMax )
    {
    this->OffsetScale->SetRange(newMin, newMax);
    modified = 1;
    }


  //
  //  Update the values of the LightboxWidthEntry and LightboxHeightEntry
  // to match the state of the viewer....
  // int wid = this->GetLightboxWidthEntry->GetValueAsInt();
  // int hit = this->GetLightboxHeightEntry->GetValueAsInt ();
  // ....compare with node value.... if different, then set.
  // this->GetLightboxWidthEntry->SetValueAsInt();
  // this->GetLightboxHeightEntry->SetValueAsInt();
  
  //
  // Update the VisibilityButton in the SliceController to match the logic state
  //
  if ( this->SliceNode->GetSliceVisible() > 0 ) 
      {
      this->GetVisibilityToggle()->SetImageToIcon ( 
            this->GetVisibilityIcons()->GetVisibleIcon ( ) );        
      } 
  else 
      {
      this->GetVisibilityToggle()->SetImageToIcon ( 
            this->GetVisibilityIcons()->GetInvisibleIcon ( ) );        
      }

  //
  // Update the Linked Controls Icon in the SliceController to match logic state.
  //
  if ( this->SliceCompositeNode != NULL && this->SliceCompositeNode->GetLinkedControl() > 0 )
    {
    this->GetLinkButton()->SetImageToIcon (
            this->GetSliceControlIcons()->GetLinkControlsIcon() );
    }
  else
    {
    this->GetLinkButton()->SetImageToIcon (
            this->GetSliceControlIcons()->GetUnlinkControlsIcon() );
    }
  
  //
  // Set the opacity value to match the value
  //
  if ( this->SliceCompositeNode != NULL && (double) this->LabelOpacityScale->GetValue() != this->SliceCompositeNode->GetLabelOpacity() )
    {
    this->LabelOpacityScale->SetValue ( this->SliceCompositeNode->GetLabelOpacity() );
    }


  //
  // Set the scale and entry widgets' value to match the offset
  //
  if ( (double) this->OffsetScale->GetValue() != this->SliceLogic->GetSliceOffset() )
    {
    this->OffsetScale->SetValue( this->SliceLogic->GetSliceOffset() );
    }


  //
  // when the composite node changes, update the menus to match
  //
  //if ( caller == this->SliceCompositeNode )
  //  {
    vtkMRMLNode *node = this->MRMLScene->GetNodeByID( this->SliceCompositeNode->GetForegroundVolumeID() );
    if ( node )
      {
      this->ForegroundSelector->SetSelected(node);
      }
    else
      {
      this->ForegroundSelector->GetWidget()->GetWidget()->GetMenu()->SelectItem("None");
      }

    node = this->MRMLScene->GetNodeByID( this->SliceCompositeNode->GetBackgroundVolumeID() );
    if ( node )
      {
      this->BackgroundSelector->SetSelected(node);
      }
    else
      {
      this->BackgroundSelector->GetWidget()->GetWidget()->GetMenu()->SelectItem("None");
      }    

    node = this->MRMLScene->GetNodeByID( this->SliceCompositeNode->GetLabelVolumeID() );
    if ( node )
      {
      this->LabelSelector->SetSelected(node);
      }
    else
      {
      this->LabelSelector->GetWidget()->GetWidget()->GetMenu()->SelectItem("None");
      }
    //}

  //
  //  Trigger events if needed
  //
  if ( modified )
    {
    this->Modified();
    }
}

//----------------------------------------------------------------------------
void vtkSlicerSliceControllerWidget::Shrink() 
{ 
  if (this->ContainerFrame && this->ContainerFrame->IsPacked())
    {
    if (this->ColorCodeButton)
      {
      this->ColorCodeButton->SetImageToPredefinedIcon (vtkKWIcon::IconSpinDown );
      this->ColorCodeButton->SetCommand (this, "Expand");
      }
    this->Script ("pack forget %s", 
                  this->ContainerFrame->GetWidgetName());
    this->InvokeEvent(vtkSlicerSliceControllerWidget::ShrinkEvent, NULL);
    }
}

//----------------------------------------------------------------------------
void vtkSlicerSliceControllerWidget::Expand() 
{ 
  if (this->ContainerFrame && !this->ContainerFrame->IsPacked())
    {
    if (this->ColorCodeButton)
      {
      this->ColorCodeButton->SetImageToPredefinedIcon (vtkKWIcon::IconSpinUp );
      this->ColorCodeButton->SetCommand (this, "Shrink");
      }
    this->Script ("pack %s -side bottom -expand 1 -fill x", 
                  this->ContainerFrame->GetWidgetName());
    this->InvokeEvent(vtkSlicerSliceControllerWidget::ExpandEvent, NULL);
    }
}

//----------------------------------------------------------------------------
void vtkSlicerSliceControllerWidget::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  // widgets?
}

