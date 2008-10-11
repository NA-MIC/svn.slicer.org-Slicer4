// .NAME vtkSlicerViewControlGUI
// .SECTION Description
// Main Data GUI and mediator methods for
// ViewControl GUI Panel in slicer3.

#ifndef __vtkSlicerViewControlGUI_h
#define __vtkSlicerViewControlGUI_h

#include "vtkObserverManager.h"
#include "vtkImageData.h"
#include "vtkTransform.h"
#include "vtkOutlineSource.h"
#include "vtkPolyDataMapper.h"
#include "vtkImageMapper.h"
#include "vtkActor2D.h"
#include "vtkActor.h"
#include "vtkFollower.h"

#include "vtkSlicerImageCloseUp2D.h"
#include "vtkSlicerImageCrossHair2D.h"
#include "vtkSlicerBaseGUIWin32Header.h"
#include "vtkSlicerComponentGUI.h"
#include "vtkSlicerInteractorStyle.h"
#include "vtkSlicerViewerInteractorStyle.h"

#include "vtkMRMLViewNode.h"
#include "vtkMRMLSliceNode.h"
#include "vtkMRMLSelectionNode.h"
#include "vtkMRMLCameraNode.h"

class vtkKWFrame;
class vtkKWPushButton;
class vtkKWTopLevel;
class vtkKWRenderWidget;

class vtkSlicerApplicationGUI;
class vtkKWScale;
class vtkKWScaleWithEntry;
class vtkKWPushButton;
class vtkKWCheckButton;
class vtkKWRadioButton;
class vtkKWLabel;
class vtkKWMenuButton;
class vtkKWEntry;
class vtkKWEntryWithLabel;
class vtkSlicerViewControlIcons;
class vtkCallbackCommand;


//BTX
#ifndef vtkSetAndObserveMRMLNodeMacro
#define vtkSetAndObserveMRMLNodeMacro(node,value)  { \
  vtkObject *oldNode = (node); \
  this->MRMLObserverManager->SetAndObserveObject ( vtkObjectPointer( &(node) ), (value) ); \
  if ( oldNode != (node) ) \
    { \
    this->InvokeEvent (vtkCommand::ModifiedEvent); \
    } \
};
#endif

#ifndef vtkSetAndObserveMRMLNodeEventsMacro
#define vtkSetAndObserveMRMLNodeEventsMacro(node,value,events)  { \
  vtkObject *oldNode = (node); \
  this->MRMLObserverManager->SetAndObserveObjectEvents ( vtkObjectPointer( &(node)), (value), (events)); \
  if ( oldNode != (node) ) \
    { \
    this->InvokeEvent (vtkCommand::ModifiedEvent); \
    } \
};
#endif
//ETX

// Description:
// This class implements Slicer's 3DView Control Panel on Slicer's main GUI Panel.
//
class VTK_SLICER_BASE_GUI_EXPORT vtkSlicerViewControlGUI : public vtkSlicerComponentGUI
{
 public:
  // Description:
  // Usual vtk class functions
  static vtkSlicerViewControlGUI* New ( );
  vtkTypeRevisionMacro ( vtkSlicerViewControlGUI, vtkSlicerComponentGUI );
  void PrintSelf ( ostream& os, vtkIndent indent );

  // Description:
  // Get/Set when a render is pending.
  vtkGetMacro ( NavigationRenderPending, int );
  vtkSetMacro ( NavigationRenderPending, int );
  // Description:
  // Get/Set when a zoom is pending.
  vtkGetMacro ( ZoomRenderPending, int );
  vtkSetMacro ( ZoomRenderPending, int );

  vtkGetMacro (EntryUpdatePending, int );
  vtkSetMacro (EntryUpdatePending, int);
  
  // Description:
  // parameters used for animated rock
  vtkGetMacro ( RockCount, int );
  vtkSetMacro ( RockCount, int );
  vtkGetMacro ( SliceMagnification, double );  
  vtkGetMacro ( SliceInteracting, int );

  vtkGetMacro (ProcessingMRMLEvent, int);
  vtkGetMacro (NavigationZoomWidgetWid, int);
  vtkGetMacro (NavigationZoomWidgetHit, int);
  
  // Description:
  // Icons that modify the widgets in ViewControlGUI
  vtkGetObjectMacro ( SlicerViewControlIcons, vtkSlicerViewControlIcons );

  // Description:
  // Get the widgets in the ViewControlGUI
  vtkGetObjectMacro (SpinButton, vtkKWCheckButton);
  vtkGetObjectMacro (RockButton, vtkKWCheckButton);
  vtkGetObjectMacro (OrthoButton, vtkKWPushButton);
  vtkGetObjectMacro (CenterButton, vtkKWPushButton);
  vtkGetObjectMacro (SelectViewButton, vtkKWMenuButton);
  vtkGetObjectMacro (SelectCameraButton, vtkKWMenuButton);
  vtkGetObjectMacro (StereoButton, vtkKWMenuButton);
  vtkGetObjectMacro (VisibilityButton, vtkKWMenuButton );
  vtkGetObjectMacro (RedFOVEntry, vtkKWEntryWithLabel);
  vtkGetObjectMacro (YellowFOVEntry, vtkKWEntry);
  vtkGetObjectMacro (GreenFOVEntry, vtkKWEntry);
  vtkGetObjectMacro (ZoomEntry, vtkKWEntryWithLabel);
  vtkGetObjectMacro (LookFromButton, vtkKWRadioButton);
  vtkGetObjectMacro (RotateAroundButton, vtkKWRadioButton );

  
  // Description:
  // Get the Widgets that display the RotateAround rollover images
  // and the LookFrom rollover images in the ViewControlGUI
  vtkGetObjectMacro (ViewAxisAIconButton, vtkKWLabel );
  vtkGetObjectMacro (ViewAxisPIconButton, vtkKWLabel );
  vtkGetObjectMacro (ViewAxisRIconButton, vtkKWLabel );
  vtkGetObjectMacro (ViewAxisLIconButton, vtkKWLabel );
  vtkGetObjectMacro (ViewAxisSIconButton, vtkKWLabel );
  vtkGetObjectMacro (ViewAxisIIconButton, vtkKWLabel );
  vtkGetObjectMacro (ViewAxisCenterIconButton, vtkKWLabel );
  vtkGetObjectMacro (ViewAxisTopCornerIconButton, vtkKWLabel );
  vtkGetObjectMacro (ViewAxisBottomCornerIconButton, vtkKWLabel);

  vtkGetObjectMacro (NavigationWidget, vtkKWRenderWidget );
  vtkGetObjectMacro (ZoomWidget, vtkKWRenderWidget );
  vtkGetObjectMacro (NavigationZoomFrame, vtkKWFrame );
  
  // Description:
  // Box that represents the 3DViewer's window in the
  // Navigation rendered view
  vtkGetObjectMacro (FOVBox, vtkOutlineSource );
  vtkSetObjectMacro (FOVBox, vtkOutlineSource );
  vtkGetObjectMacro (FOVBoxMapper, vtkPolyDataMapper);
  vtkGetObjectMacro (FOVBoxActor, vtkFollower);

  // Description:
  // Get the Widgets that display the Zoom images
  // and cursor in the ViewControlGUI
  vtkGetObjectMacro (SliceMagnifier, vtkSlicerImageCloseUp2D);
  vtkGetObjectMacro (SliceMagnifierCursor, vtkSlicerImageCrossHair2D);
  vtkGetObjectMacro (SliceMagnifierMapper, vtkImageMapper);
  vtkGetObjectMacro (SliceMagnifierActor, vtkActor2D);

  // Description:
  // Get the main slicer application
  vtkGetObjectMacro (ApplicationGUI, vtkSlicerApplicationGUI );
  virtual void SetApplicationGUI ( vtkSlicerApplicationGUI *appGUI );

  // Description:
  // API for getting & setting SliceGUI and MainViewer's interactor style
  vtkSetObjectMacro ( RedSliceEvents, vtkSlicerInteractorStyle );
  vtkSetObjectMacro ( YellowSliceEvents, vtkSlicerInteractorStyle );
  vtkSetObjectMacro ( GreenSliceEvents, vtkSlicerInteractorStyle );
  vtkSetObjectMacro ( MainViewerEvents, vtkSlicerViewerInteractorStyle );
  vtkGetObjectMacro ( RedSliceEvents, vtkSlicerInteractorStyle );
  vtkGetObjectMacro ( YellowSliceEvents, vtkSlicerInteractorStyle );
  vtkGetObjectMacro ( GreenSliceEvents, vtkSlicerInteractorStyle );
  vtkGetObjectMacro ( MainViewerEvents, vtkSlicerViewerInteractorStyle);

  // Description:
  // Methods for Get/Set the MRMLViewNode (probably
  // will be modified or phased out), or refactored into
  // an accompanying Logic class.
  vtkGetObjectMacro ( ViewNode, vtkMRMLViewNode );
  vtkSetObjectMacro ( ViewNode, vtkMRMLViewNode );
  vtkGetObjectMacro ( RedSliceNode, vtkMRMLSliceNode );
  vtkGetObjectMacro ( YellowSliceNode, vtkMRMLSliceNode );  
  vtkGetObjectMacro ( GreenSliceNode, vtkMRMLSliceNode );
  vtkMRMLViewNode *GetActiveView();
  vtkMRMLCameraNode *GetActiveCamera();
    
  // Description:
  // This method builds the Data module's GUI
  virtual void BuildGUI ( vtkKWFrame *appF );
  virtual void TearDownGUI  ( );
  
  // Description:
  // Add/Remove observers on widgets in the GUI
  virtual void AddGUIObservers ( );
  virtual void RemoveGUIObservers ( );

  // Description:
  // Add and remove observers on the
  // MainViewer so that we can manage
  // functionality of the Navigation widget
  void UpdateMainViewerInteractorStyles ( );
  void AddMainViewerEventObservers();
  void RemoveMainViewerEventObservers();
  
  // Description:
  // Add and remove observers on the
  // slice GUIs so that we can manage
  // functionality of the Zoom widget
  virtual void UpdateSliceGUIInteractorStyles();
  virtual void AddSliceEventObservers();
  virtual void RemoveSliceEventObservers();
  
  // Description:
  // Methods to update GUI, View and MRML
  virtual void RequestFOVEntriesUpdate ( );
  virtual void FOVEntriesUpdate();
  virtual void UpdateViewFromMRML();
  virtual void UpdateSlicesFromMRML();
  virtual void UpdateFromMRML ( );
  
  // Description:
  // Renders the Navigation/Zoom widget fresh
  // when scene has been modified, or
  // when view is changed.
  virtual void RequestNavigationRender ( );
  virtual void NavigationRender ( );
  virtual void RequestZoomRender ( );
  virtual void ZoomRender ( );
  
  // Description:
  // Class's mediator methods for processing events invoked by
  // either the Logic, MRML or GUI.
  virtual void ProcessLogicEvents ( vtkObject *caller, unsigned long event, void *callData );
  virtual void ProcessGUIEvents ( vtkObject *caller, unsigned long event, void *callData );
  virtual void ProcessMRMLEvents ( vtkObject *caller, unsigned long event, void *callData );
    
  // Description:
  // Describe the behavior at module enter and exit.
  virtual void Enter ( );
  virtual void Exit ( );

  // Description:
  // manages the fiducial visibility across ViewControlGUI,
  // SliceControlGUI and FiducialsGUI
  virtual void SetMRMLFiducialPointVisibility ( int state);
  virtual void SetMRMLFiducialLabelVisibility ( int state);

  // Description:
  // Starts and stops automatic view spinning
  virtual void MainViewSpin ( );
  virtual void SpinView (int direction, double degrees );
  // Description:
  // Zooms the main view with user-specified
  // relative (percent) zoom factor.
  virtual void MainViewZoom ( double factor );
  // Description:
  // Resets focal point to origin
  virtual void MainViewResetFocalPoint ( );
  // Description:
  // Sets cameras focal point.
  virtual void MainViewSetFocalPoint ( double x, double y, double z);
  // Description:
  // Starts and stops automatic view rocking
  virtual void MainViewRock ( );
  virtual void RockView ( );
  // Description:
  // Moves camera down selected axis in MainViewer and looks at
  // focal point from there
  virtual void MainViewLookFrom ( const char *dir );
  // Description:
  // Rotates camera about selected axis by an increment in MainViewer
  virtual void MainViewRotateAround ( int axis );
  // Description:
  // Sets either Parallel or Perspective Projection in MainViewer
  virtual void MainViewSetProjection ( );
  // Description:
  // Sets stereo options or turns off stereo, if
  // stereo is enabled in MainViewer
  virtual void MainViewSetStereo ( );
  // Description:
  // Sets background color in the MainViewer
  virtual void MainViewBackgroundColor ( double *color );
  // Description:
  // Toggles visibility of actors in the MainViewer
  virtual void MainViewVisibility( );

  // Description:
  // Creates the magnified slice view in ZoomWidget
  // when mouse moves over a slice window.
  virtual void SliceViewMagnify( int event, vtkSlicerInteractorStyle *istyle);

  virtual void FitFOVToBackground( double fov, int viewer );

  // Description:
  // Keeps the actors added to the Navigation Widget's
  // renderer the same as those in the 3DView's renderer.
  virtual void UpdateNavigationWidgetViewActors ( );

  // Description:
  // Updates the Navigation widget's camera to track
  // the 3DView camera
  virtual void ConfigureNavigationWidgetRender ( );

  // Description:
  // Configures the Navigation widget's camera on
  // startup.
  virtual void InitializeNavigationWidgetCamera ( );

  // Description: Similar to vtkRenderer's ResetCamera
  // method, but tries to use a smaller ViewAngle to
  // compute a nearer camera distance that still
  // accommodates all visible actors in scene but fills
  // more of the little render window with stuff. Not
  // working any differently than ResetCamera yet,
  // for reasons unknown... 
  virtual void ResetNavigationCamera ( );

  // Description:
  // Create the rectangle on the NavigationWidget that
  // indicates the 3DView's FOV.
  virtual void CreateFieldOfViewBoxActor ( );
  
  // Description:
  // Groups of callbacks that handle the state change of
  // rollover images in the ViewControlFrame. These
  // callbacks only update the visual behavior of the GUI,
  // but don't impact the application state at all.
  void ViewControlACallback ( );
  void ViewControlPCallback ( );
  void ViewControlSCallback ( );
  void ViewControlICallback ( );
  void ViewControlLCallback ( );
  void ViewControlRCallback ( );
  void EnterViewAxisACallback ( );
  void LeaveViewAxisACallback ( );
  void EnterViewAxisPCallback ( );
  void LeaveViewAxisPCallback ( );
  void EnterViewAxisRCallback ( );
  void LeaveViewAxisRCallback ( );    
  void EnterViewAxisLCallback ( );
  void LeaveViewAxisLCallback ( );
  void EnterViewAxisSCallback ( );
  void LeaveViewAxisSCallback ( );
  void EnterViewAxisICallback ( );
  void LeaveViewAxisICallback ( );

  // Description:
  // Assigns <enter> and <leave> bindings for rollover images.
  void MakeViewControlRolloverBehavior ( );

  // Description:
  // Builds pulldown menus for GUI menubuttons
  void BuildCameraSelectMenu ( );
  void BuildViewSelectMenu ( );
  void BuildStereoSelectMenu ( );
  void BuildVisibilityMenu ( );

  // Description:
  // Methods for unpacking and packing the
  // Navigation and Zoom widgets into same parcel.
  virtual void PackNavigationWidget ( );
  virtual void PackZoomWidget ( ) ;

 protected:
  vtkSlicerViewControlGUI ( );
  virtual ~vtkSlicerViewControlGUI ( );
    
  int NavigationRenderPending;
  int ZoomRenderPending;
  int  EntryUpdatePending;
  int ProcessingMRMLEvent;
  bool SceneClosing;
  
  vtkSlicerApplicationGUI *ApplicationGUI;
  vtkSlicerViewControlIcons *SlicerViewControlIcons;
  vtkKWCheckButton *SpinButton;
  vtkKWCheckButton *RockButton;
  vtkKWPushButton *OrthoButton;
  vtkKWRadioButton *LookFromButton;
  vtkKWRadioButton *RotateAroundButton;
  vtkKWPushButton *CenterButton;
  vtkKWMenuButton *StereoButton;
  vtkKWMenuButton *SelectViewButton;
  vtkKWMenuButton *SelectCameraButton;
  vtkKWMenuButton *VisibilityButton;
  vtkKWEntryWithLabel *RedFOVEntry;
  vtkKWEntry *YellowFOVEntry;
  vtkKWEntry *GreenFOVEntry;
  vtkKWEntryWithLabel *ZoomEntry;
    
  // navzoom scale, navzoomin/outiconbutton tmpNavigationZoom, all the icon buttons.    
  // Description:
  // These widgets display icons that indicate
  // zoom-in and zoom-out functionality in the
  // ViewControlFrame's Navigation widget
  vtkKWRenderWidget *NavigationWidget;
  vtkKWRenderWidget *ZoomWidget;
  vtkKWFrame *NavigationZoomFrame;

  // Description:
  // These widgets tile a composite image
  // for labeling the 3 axes (R,L,S,I,A,P)
  // The composite image displays highlighting
  // behavior during mouseover
  vtkKWLabel *ViewAxisAIconButton;
  vtkKWLabel *ViewAxisPIconButton;
  vtkKWLabel *ViewAxisRIconButton;
  vtkKWLabel *ViewAxisLIconButton;
  vtkKWLabel *ViewAxisSIconButton;
  vtkKWLabel *ViewAxisIIconButton;
  vtkKWLabel *ViewAxisCenterIconButton;    
  vtkKWLabel *ViewAxisTopCornerIconButton;
  vtkKWLabel *ViewAxisBottomCornerIconButton;

  // Description:
  // pipeline objects for accomplishing the slice zoom
  // in the Zoom widget.
  vtkSlicerImageCloseUp2D *SliceMagnifier;
  vtkSlicerImageCrossHair2D *SliceMagnifierCursor;
  vtkImageMapper *SliceMagnifierMapper;
  vtkActor2D *SliceMagnifierActor;
  
  // Description:
  // objects for building the FOV box in the
  // Navigation widget
  vtkOutlineSource *FOVBox;
  vtkPolyDataMapper *FOVBoxMapper;
  vtkFollower *FOVBoxActor;

  // Description:
  // MRML and GUI objects that this class will need
  // to set and observe.
  vtkMRMLViewNode *ViewNode;
  vtkSlicerInteractorStyle *RedSliceEvents;
  vtkSlicerInteractorStyle *YellowSliceEvents;
  vtkSlicerInteractorStyle *GreenSliceEvents;
  vtkMRMLSliceNode *RedSliceNode;
  vtkMRMLSliceNode *YellowSliceNode;
  vtkMRMLSliceNode *GreenSliceNode;
  vtkSlicerViewerInteractorStyle *MainViewerEvents;

  int RockCount;
  int NavigationZoomWidgetWid;
  int NavigationZoomWidgetHit;
  double SliceMagnification;
  int SliceInteracting;

 private:
  vtkSlicerViewControlGUI ( const vtkSlicerViewControlGUI& ); // Not implemented.
  void operator = ( const vtkSlicerViewControlGUI& ); //Not implemented.
};


#endif
