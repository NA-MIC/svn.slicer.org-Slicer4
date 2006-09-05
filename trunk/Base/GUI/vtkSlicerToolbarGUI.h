// .NAME vtkSlicerToolbarGUI
// .SECTION Description
// Main Data GUI and mediator methods for slicer3.

#ifndef __vtkSlicerToolbarGUI_h
#define __vtkSlicerToolbarGUI_h

#include "vtkSlicerBaseGUIWin32Header.h"

#include "vtkSlicerComponentGUI.h"
#include "vtkSlicerToolbarIcons.h"

#include "vtkKWFrame.h"
#include "vtkKWToolbar.h"
#include "vtkKWPushButton.h"

class vtkSlicerApplicationGUI;

// Description:
// This class implements Slicer's Application Toolbar 
//
class VTK_SLICER_BASE_GUI_EXPORT vtkSlicerToolbarGUI : public vtkSlicerComponentGUI
{
 public:
    // Description:
    // Usual vtk class functions
    static vtkSlicerToolbarGUI* New ( );
    vtkTypeRevisionMacro ( vtkSlicerToolbarGUI, vtkSlicerComponentGUI );
    void PrintSelf ( ostream& os, vtkIndent indent );
    
    // Description:
    // Get the class containing all slicer GUI images for logos/icons
    vtkGetObjectMacro (SlicerToolbarIcons, vtkSlicerToolbarIcons);

    // Description:
    // Get the main slicer toolbars.
    vtkGetObjectMacro (ApplicationGUI, vtkSlicerApplicationGUI );
    virtual void SetApplicationGUI ( vtkSlicerApplicationGUI *appGUI );
    
    vtkGetObjectMacro (ModulesToolbar, vtkKWToolbar);
    vtkGetObjectMacro (LoadSaveToolbar, vtkKWToolbar );
    vtkGetObjectMacro (ViewToolbar, vtkKWToolbar);
    vtkGetObjectMacro (MouseModeToolbar, vtkKWToolbar);


    // Description:
    // Get the widgets that display the toolbar icons
    vtkGetObjectMacro (HomeIconButton, vtkKWPushButton); 
    vtkGetObjectMacro (DataIconButton, vtkKWPushButton);
    vtkGetObjectMacro (VolumeIconButton, vtkKWPushButton );
    vtkGetObjectMacro (ModelIconButton, vtkKWPushButton );
    vtkGetObjectMacro (EditorIconButton, vtkKWPushButton );
    vtkGetObjectMacro (EditorToolboxIconButton, vtkKWPushButton );
    vtkGetObjectMacro (TransformIconButton, vtkKWPushButton );
    vtkGetObjectMacro (ColorIconButton, vtkKWPushButton );
    vtkGetObjectMacro (FiducialsIconButton, vtkKWPushButton);
    vtkGetObjectMacro (MeasurementsIconButton, vtkKWPushButton);
    vtkGetObjectMacro (SaveSceneIconButton, vtkKWPushButton );
    vtkGetObjectMacro (LoadSceneIconButton, vtkKWPushButton );
    vtkGetObjectMacro (ConventionalViewIconButton, vtkKWPushButton );    
    vtkGetObjectMacro (OneUp3DViewIconButton, vtkKWPushButton );
    vtkGetObjectMacro (OneUpSliceViewIconButton, vtkKWPushButton );
    vtkGetObjectMacro (FourUpViewIconButton, vtkKWPushButton );
    vtkGetObjectMacro (Tabbed3DViewIconButton, vtkKWPushButton );
    vtkGetObjectMacro (TabbedSliceViewIconButton, vtkKWPushButton );
    vtkGetObjectMacro (LightBoxViewIconButton, vtkKWPushButton );
    vtkGetObjectMacro (MousePickIconButton, vtkKWPushButton );
    vtkGetObjectMacro (MousePanIconButton, vtkKWPushButton );
    vtkGetObjectMacro (MouseRotateIconButton, vtkKWPushButton );
    vtkGetObjectMacro (MouseZoomIconButton, vtkKWPushButton );
    vtkGetObjectMacro (MousePlaceFiducialIconButton, vtkKWPushButton );

    // Description:
    // This method builds the Data module's GUI
    virtual void BuildGUI ( ) ;

    // Description:
    // Add/Remove observers on widgets in the GUI
    virtual void AddGUIObservers ( );
    virtual void RemoveGUIObservers ( );

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
    
 protected:
    vtkSlicerToolbarGUI ( );
    virtual ~vtkSlicerToolbarGUI ( );

    // Description:
    // Main Application Toolbars
    vtkKWToolbar *ModulesToolbar;
    vtkKWToolbar *LoadSaveToolbar;
    vtkKWToolbar *ViewToolbar;
    vtkKWToolbar *MouseModeToolbar;

    // Description:
    // Contains icons
    vtkSlicerToolbarIcons *SlicerToolbarIcons;
    
    // Description:
    // Widgets that display toolbar icons
    vtkKWPushButton *HomeIconButton;
    vtkKWPushButton *DataIconButton;
    vtkKWPushButton *VolumeIconButton;
    vtkKWPushButton *ModelIconButton;
    vtkKWPushButton *EditorIconButton;
    vtkKWPushButton *EditorToolboxIconButton;
    vtkKWPushButton *TransformIconButton;    
    vtkKWPushButton *ColorIconButton;
    vtkKWPushButton *FiducialsIconButton;
    vtkKWPushButton *MeasurementsIconButton;
    vtkKWPushButton *SaveSceneIconButton;
    vtkKWPushButton *LoadSceneIconButton;
    vtkKWPushButton *ConventionalViewIconButton;
    vtkKWPushButton *OneUp3DViewIconButton;
    vtkKWPushButton *OneUpSliceViewIconButton;
    vtkKWPushButton *FourUpViewIconButton;
    vtkKWPushButton *Tabbed3DViewIconButton;
    vtkKWPushButton *TabbedSliceViewIconButton;
    vtkKWPushButton *LightBoxViewIconButton;
    vtkKWPushButton *MousePickIconButton;
    vtkKWPushButton *MousePlaceFiducialIconButton;
    vtkKWPushButton *MousePanIconButton;
    vtkKWPushButton *MouseRotateIconButton;
    vtkKWPushButton *MouseZoomIconButton;
    
    vtkSlicerApplicationGUI *ApplicationGUI;
    
 private:
    vtkSlicerToolbarGUI ( const vtkSlicerToolbarGUI& ); // Not implemented.
    void operator = ( const vtkSlicerToolbarGUI& ); //Not implemented.
};


#endif
