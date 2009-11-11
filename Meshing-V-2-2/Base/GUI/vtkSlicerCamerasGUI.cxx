#include "vtkObject.h"
#include "vtkObjectFactory.h"
#include "vtkCommand.h"

#include "vtkSlicerCamerasGUI.h"
#include "vtkSlicerApplication.h"
#include "vtkSlicerModuleCollapsibleFrame.h"
#include "vtkMRMLCameraNode.h"
#include "vtkSlicerNodeSelectorWidget.h"

#include "vtkKWWidget.h"
#include "vtkKWCheckButton.h"
#include "vtkKWMenuButton.h"
#include "vtkKWFrameWithLabel.h"
#include "vtkKWFrame.h"

//---------------------------------------------------------------------------
vtkStandardNewMacro (vtkSlicerCamerasGUI );
vtkCxxRevisionMacro ( vtkSlicerCamerasGUI, "$Revision: 1.0 $");

//----------------------------------------------------------------------------
class vtkSlicerCamerasGUIInternals
{
public:
  vtksys_stl::string ScheduleUpdateCameraSelectorTimerId;
};

//---------------------------------------------------------------------------
vtkSlicerCamerasGUI::vtkSlicerCamerasGUI ( )
{
  this->Internals = new vtkSlicerCamerasGUIInternals;

  this->ViewSelectorWidget = NULL;
    this->CameraSelectorWidget = NULL;
}

//---------------------------------------------------------------------------
vtkSlicerCamerasGUI::~vtkSlicerCamerasGUI ( )
{
  this->RemoveGUIObservers();
    
  if (this->ViewSelectorWidget)
    {
    this->ViewSelectorWidget->SetParent(NULL);
    this->ViewSelectorWidget->Delete();
    }

  if (this->CameraSelectorWidget)
    {
    this->CameraSelectorWidget->SetParent(NULL );
    this->CameraSelectorWidget->Delete ( );
    }

  delete this->Internals;
  this->Internals = NULL;
    }

//---------------------------------------------------------------------------
void vtkSlicerCamerasGUI::PrintSelf ( ostream& os, vtkIndent indent )
{
    this->vtkObject::PrintSelf ( os, indent );

    os << indent << "SlicerCamerasGUI: " << this->GetClassName ( ) << "\n";
}

//---------------------------------------------------------------------------
void vtkSlicerCamerasGUI::RemoveGUIObservers ( )
{
  this->ViewSelectorWidget->RemoveObservers(
    vtkSlicerNodeSelectorWidget::NodeSelectedEvent, 
                                            (vtkCommand *)this->GUICallbackCommand );

  this->ViewSelectorWidget->RemoveObservers(
    vtkSlicerNodeSelectorWidget::NodeAddedEvent, 
    (vtkCommand *)this->GUICallbackCommand);

  this->CameraSelectorWidget->RemoveObservers(
    vtkSlicerNodeSelectorWidget::NodeSelectedEvent, 
    (vtkCommand *)this->GUICallbackCommand);

  this->CameraSelectorWidget->RemoveObservers(
    vtkSlicerNodeSelectorWidget::NodeAddedEvent, 
    (vtkCommand *)this->GUICallbackCommand );

  this->GetMRMLScene()->RemoveObservers(
    vtkMRMLScene::NodeAddedEvent, 
                                             (vtkCommand *)this->GUICallbackCommand );

  this->GetMRMLScene()->RemoveObservers(
    vtkMRMLScene::NodeRemovedEvent, 
    (vtkCommand *)this->GUICallbackCommand );
}

//---------------------------------------------------------------------------
void vtkSlicerCamerasGUI::AddGUIObservers ( )
{
  this->ViewSelectorWidget->AddObserver(
    vtkSlicerNodeSelectorWidget::NodeSelectedEvent, 
    (vtkCommand *)this->GUICallbackCommand );

  this->ViewSelectorWidget->AddObserver(
    vtkSlicerNodeSelectorWidget::NodeAddedEvent, 
    (vtkCommand *)this->GUICallbackCommand );

  this->CameraSelectorWidget->AddObserver(
    vtkSlicerNodeSelectorWidget::NodeSelectedEvent, 
                                        (vtkCommand *)this->GUICallbackCommand );

  this->CameraSelectorWidget->AddObserver(
    vtkSlicerNodeSelectorWidget::NodeAddedEvent, 
                                             (vtkCommand *)this->GUICallbackCommand );

  // Listen to the scene so that we can listen to camera node when their
  // ActiveTag is modified.

  this->GetMRMLScene()->AddObserver(
    vtkMRMLScene::NodeAddedEvent, 
    (vtkCommand *)this->GUICallbackCommand );

  this->GetMRMLScene()->AddObserver(
    vtkMRMLScene::NodeRemovedEvent, 
    (vtkCommand *)this->GUICallbackCommand );

  // Listen to camera node that have been created before we were even created

  std::vector<vtkMRMLNode*> snodes;
  int nnodes = 
    this->GetMRMLScene()->GetNodesByClass("vtkMRMLCameraNode", snodes);
  for (int n = 0; n < nnodes; n++)
    {
    vtkMRMLCameraNode *node = vtkMRMLCameraNode::SafeDownCast(snodes[n]);
    node->AddObserver(vtkMRMLCameraNode::ActiveTagModifiedEvent, 
                      this->GUICallbackCommand);
    }
}

//---------------------------------------------------------------------------
void vtkSlicerCamerasGUI::ProcessGUIEvents(
  vtkObject *caller,
                                             unsigned long event, void *callData )
{
  if (this->ViewSelectorWidget == 
      vtkSlicerNodeSelectorWidget::SafeDownCast(caller))
    {
    if (event == vtkSlicerNodeSelectorWidget::NodeAddedEvent)
      {
      vtkMRMLViewNode *added_view_node = 
        vtkMRMLViewNode::SafeDownCast((vtkObject*)callData);
      if (added_view_node)
        {
        added_view_node->SetActive(1);
        }
      }
    else if (event == vtkSlicerNodeSelectorWidget::NodeSelectedEvent)
      {
      this->UpdateCameraSelector();
      }
      }

  if (this->CameraSelectorWidget == 
      vtkSlicerNodeSelectorWidget::SafeDownCast(caller))
    {
    if (event == vtkSlicerNodeSelectorWidget::NodeAddedEvent)
      {
      // vtkMRMLCameraNode *calldata_camera_node = 
      // vtkMRMLCameraNode::SafeDownCast((vtkObject*)callData);
    }
    else if (event == vtkSlicerNodeSelectorWidget::NodeSelectedEvent)
      {
      vtkMRMLCameraNode *selected_camera_node = 
        vtkMRMLCameraNode::SafeDownCast(
          this->CameraSelectorWidget->GetSelected());
      if (selected_camera_node)
        {
        vtkMRMLViewNode *selected_view_node = 
          vtkMRMLViewNode::SafeDownCast(
            this->ViewSelectorWidget->GetSelected());
        if (selected_view_node)
    {
          selected_camera_node->SetActiveTag(selected_view_node->GetID());
          }
        }
      }
    }

  // Listen to the scene so that we can listen to camera node when their
  // ActiveTag is modified.

  if (this->GetMRMLScene() == vtkMRMLScene::SafeDownCast(caller))
    {
    vtkMRMLNode *node = (vtkMRMLNode*) (callData);
    if (node != NULL && node->IsA("vtkMRMLCameraNode"))
      {
      if (event == vtkMRMLScene::NodeAddedEvent)
      {
        node->AddObserver(vtkMRMLCameraNode::ActiveTagModifiedEvent, 
                          this->GUICallbackCommand);
        }
      else if (event == vtkMRMLScene::NodeRemovedEvent)
        {
        node->RemoveObservers(vtkMRMLCameraNode::ActiveTagModifiedEvent, 
                              this->GUICallbackCommand);
        }
      }
    }

  // ActiveTag modified, update the menus...

  vtkMRMLCameraNode *cam_node = vtkMRMLCameraNode::SafeDownCast(caller);
  if (cam_node && event == vtkMRMLCameraNode::ActiveTagModifiedEvent)
    {
    // Call UpdateCameraSelector asynchronously. We do not want to do that
    // while ActiveTag are being reshuffled, since we may call
    // this->CameraSelectorWidget->SetSelected and there is no way to prevent
    // it from invoking an event and changing an ActiveTag... 
    this->ScheduleUpdateCameraSelector();
    }
} 

//---------------------------------------------------------------------------
void vtkSlicerCamerasGUI::UpdateCameraSelector()
{
  this->CameraSelectorWidget->UpdateMenu();

  // Update the camera selector to reflect which 
  // camera the selected view is using.

  if (this->ViewSelectorWidget)
    {
    vtkMRMLViewNode *selected_view_node = 
      vtkMRMLViewNode::SafeDownCast(this->ViewSelectorWidget->GetSelected());
    if (selected_view_node)
      {
      vtkMRMLCameraNode *found_camera_node = NULL;
      std::vector<vtkMRMLNode*> snodes;
      int nnodes = this->GetMRMLScene()->GetNodesByClass(
        "vtkMRMLCameraNode", snodes);
      for (int n = 0; n < nnodes; n++)
        {
        vtkMRMLCameraNode *camera_node = 
          vtkMRMLCameraNode::SafeDownCast(snodes[n]);
        if (camera_node && 
            camera_node->GetActiveTag() && 
            !strcmp(camera_node->GetActiveTag(), selected_view_node->GetID()))
          {
          found_camera_node = camera_node;
          break;
          }
        }
      this->CameraSelectorWidget->SetSelected(found_camera_node);
      }
    }
}

//----------------------------------------------------------------------------
void vtkSlicerCamerasGUI::ScheduleUpdateCameraSelector()
{
  // Already scheduled

  if (this->Internals->ScheduleUpdateCameraSelectorTimerId.size())
{
    return;
    }

  this->Internals->ScheduleUpdateCameraSelectorTimerId =
    this->Script(
    "after 500 {catch {%s UpdateCameraSelectorCallback}}", this->GetTclName());
}

//----------------------------------------------------------------------------
void vtkSlicerCamerasGUI::UpdateCameraSelectorCallback()
{
  if (!this->GetApplication() || this->GetApplication()->GetInExit())
    {
    return;
    }

  this->UpdateCameraSelector();
  this->Internals->ScheduleUpdateCameraSelectorTimerId = "";
}

//---------------------------------------------------------------------------
void vtkSlicerCamerasGUI::UpdateViewSelector()
{
  this->ViewSelectorWidget->UpdateMenu();
}

//---------------------------------------------------------------------------
void vtkSlicerCamerasGUI::ProcessMRMLEvents(
  vtkObject *caller,
                                              unsigned long event, void *callData )
{
    // Fill in
}

//---------------------------------------------------------------------------
void vtkSlicerCamerasGUI::Enter ( )
{
    // Fill in
}

//---------------------------------------------------------------------------
void vtkSlicerCamerasGUI::Exit ( )
{
    // Fill in
}

//---------------------------------------------------------------------------
void vtkSlicerCamerasGUI::BuildGUI ( )
{
  vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
  // Define your help text here.
  const char *help = "**Cameras Module:** Create new views and cameras. The view pulldown menu below can be used to create new views and select the active view. Switch the layout to \"Tabbed 3D Layout\" from the layout icon in the toolbar to access multiple views. The view selected in \"Tabbed 3D Layout\" becomes the active view and replaces the 3D view in all other layouts. The camera pulldown menu below can be used to set the active camera for the selected view. WARNING: this is rather experimental at the moment (fiducials, IO/data, closing the scene are probably broken for new views). ";
  
  // ---
  // MODULE GUI FRAME 
  // configure a page for a camera 
  // later, switch on the modulesButton in the SlicerControlGUI
  // ---
  // create a page
  this->UIPanel->AddPage ( "Cameras", "Cameras", NULL );
  
  vtkKWWidget *page = this->UIPanel->GetPageWidget ( "Cameras" );
  
  // HELP FRAME
  vtkSlicerModuleCollapsibleFrame *cameraHelpFrame = 
    vtkSlicerModuleCollapsibleFrame::New();
  cameraHelpFrame->SetParent ( page );
  cameraHelpFrame->Create ( );
  //cameraHelpFrame->CollapseFrame();
  cameraHelpFrame->SetLabelText ("Help");
  app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                cameraHelpFrame->GetWidgetName(), page->GetWidgetName());
  
  // configure the parent classes help text widget
  this->HelpText->SetParent ( cameraHelpFrame->GetFrame() );
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
  // camera FRAME            
  vtkSlicerModuleCollapsibleFrame *cameraFrame = 
    vtkSlicerModuleCollapsibleFrame::New();
  cameraFrame->SetParent ( page );
  cameraFrame->Create ( );
  cameraFrame->SetLabelText ("Camera");
  app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                cameraFrame->GetWidgetName(), page->GetWidgetName());

  // selector for view
  this->ViewSelectorWidget = vtkSlicerNodeSelectorWidget::New() ;
  this->ViewSelectorWidget->SetParent(cameraFrame->GetFrame());
  this->ViewSelectorWidget->Create();
  this->ViewSelectorWidget->SetNodeClass(
    "vtkMRMLViewNode", NULL, NULL, NULL);
  this->ViewSelectorWidget->SetMRMLScene(this->GetMRMLScene());
  this->ViewSelectorWidget->SetNewNodeEnabled(1);
  this->ViewSelectorWidget->SetShowHidden(1);
  this->ViewSelectorWidget->SetNoneEnabled(0);
  this->ViewSelectorWidget->SetBorderWidth(2);
  this->ViewSelectorWidget->SetPadX(2);
  this->ViewSelectorWidget->SetPadY(2);
  this->ViewSelectorWidget->GetWidget()->GetWidget()->IndicatorVisibilityOff();
  this->ViewSelectorWidget->GetWidget()->GetWidget()->SetWidth(24);
  this->ViewSelectorWidget->SetLabelText("View");
  this->ViewSelectorWidget->SetBalloonHelpString(
    "select a view from the current scene.");  
  this->Script("pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
               this->ViewSelectorWidget->GetWidgetName());
  
  // selector for camera
  this->CameraSelectorWidget = vtkSlicerNodeSelectorWidget::New() ;
  this->CameraSelectorWidget->SetParent ( cameraFrame->GetFrame() );
  this->CameraSelectorWidget->Create ( );
  this->CameraSelectorWidget->SetNodeClass(
    "vtkMRMLCameraNode", NULL, NULL, NULL);
  this->CameraSelectorWidget->SetMRMLScene(this->GetMRMLScene());
  this->CameraSelectorWidget->SetNewNodeEnabled(1);
  this->CameraSelectorWidget->SetShowHidden(1);
  this->CameraSelectorWidget->SetNoneEnabled(0);
  this->CameraSelectorWidget->SetBorderWidth(2);
  this->CameraSelectorWidget->SetPadX(2);
  this->CameraSelectorWidget->SetPadY(2);
  this->CameraSelectorWidget->GetWidget()->GetWidget()->IndicatorVisibilityOff();
  this->CameraSelectorWidget->GetWidget()->GetWidget()->SetWidth(24);
  this->CameraSelectorWidget->SetLabelText( "Camera");
  this->CameraSelectorWidget->SetBalloonHelpString(
    "select a camera from the current scene.");  
  this->Script("pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
                  this->CameraSelectorWidget->GetWidgetName());

  cameraFrame->Delete();
  cameraHelpFrame->Delete();

  this->UpdateViewSelector();
  this->UpdateCameraSelector();
}





