/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkSlicerApplicationGUI.h,v $
  Date:      $Date: 2006/01/08 04:48:05 $
  Version:   $Revision: 1.45 $

=========================================================================auto=*/
// .NAME vtkSlicerApplicationGUI 
// .SECTION Description
// Main application GUI for slicer3.  Points to the ApplicationLogic and
// reflects changes in that logic back onto the UI.  Also routes changes
// from the GUI into the Logic to effect the user's desires.


#ifndef __vtkSlicerApplicationGUI_h
#define __vtkSlicerApplicationGUI_h

#include "vtkSlicerBaseGUIWin32Header.h"
#include "vtkSlicerComponentGUI.h"
#include "vtkSlicerVolumesGUI.h"
#include "vtkSlicerModelsGUI.h"
#include "vtkKWWindow.h"
#include "vtkKWMenuButton.h"
#include "vtkKWFrame.h"
#include "vtkKWRenderWidget.h"

class vtkObject;
class vtkKWPushButton;

// Description:
// This class implements Slicer's main Application GUI.
//
class VTK_SLICER_BASE_GUI_EXPORT vtkSlicerApplicationGUI : public vtkSlicerComponentGUI
{
 public:
    static vtkSlicerApplicationGUI* New (  );
    vtkTypeRevisionMacro ( vtkSlicerApplicationGUI, vtkSlicerComponentGUI );

    // Description:
    // Get/Set Macros for protected vtkSlicerApplicationGUI dimensions
    vtkGetMacro ( DefaultSlicerWindowWidth, int );
    vtkSetMacro ( DefaultSlicerWindowWidth, int );
    vtkGetMacro ( DefaultSlicerWindowHeight, int );
    vtkSetMacro ( DefaultSlicerWindowHeight, int );
    vtkGetMacro ( DefaultMainViewerWidth, int);
    vtkSetMacro ( DefaultMainViewerWidth, int);
    vtkGetMacro ( DefaultMainViewerHeight, int);
    vtkSetMacro ( DefaultMainViewerHeight, int);
    vtkGetMacro ( DefaultSliceGUIFrameHeight, int);
    vtkSetMacro ( DefaultSliceGUIFrameHeight, int);
    vtkGetMacro ( DefaultSliceGUIFrameWidth, int);
    vtkSetMacro ( DefaultSliceGUIFrameWidth, int);
    vtkGetMacro ( DefaultSliceWindowWidth, int);
    vtkSetMacro ( DefaultSliceWindowWidth, int);
    vtkGetMacro ( DefaultSliceWindowHeight, int);
    vtkSetMacro ( DefaultSliceWindowHeight, int);
    vtkGetMacro ( DefaultGUIPanelWidth, int);
    vtkSetMacro ( DefaultGUIPanelWidth, int);
    vtkGetMacro ( DefaultGUIPanelHeight, int);
    vtkSetMacro ( DefaultGUIPanelHeight, int);
    vtkGetMacro ( DefaultLogoFrameHeight, int );
    vtkSetMacro ( DefaultLogoFrameHeight, int );
    vtkGetMacro ( DefaultSlicerControlFrameHeight, int );
    vtkSetMacro ( DefaultSlicerControlFrameHeight, int );
    vtkGetMacro ( DefaultModuleControlPanelHeight, int );
    vtkSetMacro ( DefaultModuleControlPanelHeight, int );
    vtkGetMacro ( DefaultSliceControlFrameHeight, int );
    vtkSetMacro ( DefaultSliceControlFrameHeight, int );
    vtkGetMacro ( DefaultViewControlFrameHeight, int );
    vtkSetMacro ( DefaultViewControlFrameHeight, int );

    // Description:
    // These Get/Set methods for frames in the GUI panel.
    vtkGetObjectMacro ( SlicerControlFrame, vtkKWFrame );
    vtkGetObjectMacro ( SliceControlFrame, vtkKWFrame );
    vtkGetObjectMacro ( ViewControlFrame, vtkKWFrame );
    vtkGetObjectMacro ( ModulesButton, vtkKWMenuButton );
    vtkGetObjectMacro ( HomeButton, vtkKWPushButton );
    vtkGetObjectMacro ( VolumesButton, vtkKWPushButton );
    vtkGetObjectMacro ( ModelsButton, vtkKWPushButton );
    vtkGetObjectMacro ( DataButton, vtkKWPushButton );
    vtkGetObjectMacro ( DefaultSlice0Frame, vtkKWFrame );
    vtkGetObjectMacro ( DefaultSlice1Frame, vtkKWFrame );
    vtkGetObjectMacro ( DefaultSlice2Frame, vtkKWFrame );
    vtkGetObjectMacro ( MainViewer, vtkKWRenderWidget );

    // Description:
    // Get/Set the main slicer window.
    vtkGetObjectMacro ( MainSlicerWin, vtkKWWindow );
    
    // Description:
    // This method builds Slicer's main GUI
    virtual void BuildGUI ( );
    virtual void AddGUIObservers ( );
    virtual void RemoveGUIObservers ( );
    virtual void AddLogicObservers ( );
    virtual void RemoveLogicObservers ( );

    virtual void ProcessLogicEvents ( vtkObject *caller, unsigned long event,
                                            void *callData );
    virtual void ProcessGUIEvents ( vtkObject *caller, unsigned long event,
                                           void *callData );
    
    virtual void Enter ( );
    virtual void Exit ( );
    
    // Description:
    // These methods set up default dimensions for the Slicer Window
    virtual void InitDefaultGUIPanelDimensions ( );
    virtual void InitDefaultSlicePanelDimensions ( );
    virtual void InitDefaultMainViewerDimensions ( );
    virtual void InitDefaultSlicerWindowDimensions ( );

    // Description:
    // These methods configure and pack the Slicer Window
    virtual void ConfigureMainSlicerWindow ( );
    virtual void ConfigureMainViewerPanel ( );
    virtual void ConfigureSliceViewersPanel ( );
    virtual void ConfigureGUIPanel ( );
    
    // Description:
    // These methods populate the various GUI Panel frames
    virtual void BuildMainViewer ( );
    virtual void BuildLogoGUIPanel ( );
    virtual void BuildSlicerControlGUIPanel ( );
    virtual void BuildSliceControlGUIPanel ( );
    virtual void BuildViewControlGUIPanel ( );

    // Desrciption:
    // These methods delete widgets belonging to components of the Slicer Window
    virtual void DeleteGUIPanelWidgets ( );
    virtual void DeleteFrames ( );

    // Description:
    // Display Slicer's main window
    virtual void DisplayMainSlicerWindow ( );

 protected:
    vtkSlicerApplicationGUI ( );
    ~vtkSlicerApplicationGUI ( );

    // Description:
    // Widgets for the main Slicer UI panel    
    vtkKWFrame *LogoFrame;
    vtkKWFrame *SlicerControlFrame;
    vtkKWFrame *SliceControlFrame;
    vtkKWFrame *ViewControlFrame;
    vtkKWPushButton *HomeButton;
    vtkKWPushButton *DataButton;
    vtkKWPushButton *VolumesButton;
    vtkKWPushButton *ModelsButton;
    vtkKWFrame *DefaultSlice0Frame;
    vtkKWFrame *DefaultSlice1Frame;
    vtkKWFrame *DefaultSlice2Frame;
    vtkKWRenderWidget *MainViewer;
    // Description:
    // Widgets for the modules GUI panels
    vtkKWMenuButton *ModulesButton;
    // Description:
    // Main Slicer window
    vtkKWWindow *MainSlicerWin;

    // Description:
    // Dimensions for the Default Window & components
    int DefaultSlicerWindowHeight;
    int DefaultSlicerWindowWidth;
    int DefaultMainViewerHeight;
    int DefaultMainViewerWidth;
    int DefaultSliceGUIFrameHeight;
    int DefaultSliceGUIFrameWidth;
    int DefaultSliceWindowHeight;
    int DefaultSliceWindowWidth;
    int DefaultGUIPanelHeight;
    int DefaultGUIPanelWidth;
    // Description:
    // Dimensions for specific GUI panel components
    int DefaultLogoFrameHeight;
    int DefaultSlicerControlFrameHeight;
    int DefaultModuleControlPanelHeight;
    int DefaultSliceControlFrameHeight;    
    int DefaultViewControlFrameHeight;
    
 private:
    vtkSlicerApplicationGUI ( const vtkSlicerApplicationGUI& ); // Not implemented.
    void operator = ( const vtkSlicerApplicationGUI& ); //Not implemented.
}; 

#endif
