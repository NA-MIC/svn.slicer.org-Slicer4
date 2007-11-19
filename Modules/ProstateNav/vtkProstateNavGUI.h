/*=auto=========================================================================

  Portions (c) Copyright 2007 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: $
  Date:      $Date: $
  Version:   $Revision: $

=========================================================================auto=*/

#ifndef __vtkProstateNavGUI_h
#define __vtkProstateNavGUI_h

#ifndef USE_NAVITRACK
#define USE_NAVITRACK
#endif


#ifdef WIN32
#include "vtkProstateNavWin32Header.h"
#endif

#include "vtkSlicerModuleGUI.h"
#include "vtkProstateNavLogic.h"

#include "vtkIGTDataManager.h"
#include "vtkIGTPat2ImgRegistration.h"
#include "vtkCallbackCommand.h"
#include "vtkSlicerInteractorStyle.h"
#include "vtkSlicerVolumesLogic.h"

#include <string>

#ifdef USE_NAVITRACK
#include "vtkIGTOpenTrackerStream.h"

#endif
#ifdef USE_IGSTK
#include "vtkIGTIGSTKStream.h"
#endif

class vtkKWPushButton;
class vtkKWPushButtonSet;
class vtkKWEntryWithLabel;
class vtkKWMenuButtonWithLabel;
class vtkKWMenuButton;
class vtkKWCheckButton;
class vtkKWScaleWithEntry;
class vtkKWEntry;
class vtkKWFrame;
class vtkKWEntryWithLabel;
class vtkKWLoadSaveButtonWithLabel;
class vtkKWMultiColumnListWithScrollbars;
class vtkKWWizardWidget;

class vtkProstateNavStep;

// Description:    
// This class implements Slicer's Volumes GUI
//
class VTK_PROSTATENAV_EXPORT vtkProstateNavGUI : public vtkSlicerModuleGUI
{
 public:
    //BTX
    enum {
      SLICE_DRIVER_USER    = 0,
      SLICE_DRIVER_LOCATOR = 1,
      SLICE_DRIVER_RTIMAGE = 2
    };
    enum {
      SLICE_PLANE_RED    = 0,
      SLICE_PLANE_YELLOW = 1,
      SLICE_PLANE_GREEN  = 2
    };
    enum {
      SLICE_RTIMAGE_PERP      = 0,
      SLICE_RTIMAGE_INPLANE90 = 1,
      SLICE_RTIMAGE_INPLANE   = 2
    };
    //ETX


 public:
    // Description:    
    // Usual vtk class functions
    static vtkProstateNavGUI* New (  );
    vtkTypeRevisionMacro ( vtkProstateNavGUI, vtkSlicerModuleGUI );
    void PrintSelf (ostream& os, vtkIndent indent );
   
    //SendDATANavitrack
    // Description:    
    // Get methods on class members (no Set methods required)
    vtkGetObjectMacro ( Logic, vtkProstateNavLogic );

    // Description:
    // API for setting VolumeNode, VolumeLogic and
    // for both setting and observing them.
    void SetModuleLogic ( vtkProstateNavLogic *logic )
        { this->SetLogic ( vtkObjectPointer (&this->Logic), logic ); }
    void SetAndObserveModuleLogic ( vtkProstateNavLogic *logic )
        { this->SetAndObserveLogic ( vtkObjectPointer (&this->Logic), logic ); }
    // Description: 
    // Get wizard widget
    vtkGetObjectMacro(WizardWidget, vtkKWWizardWidget);

    // Description:    
    // This method builds the IGTDemo module GUI
    virtual void BuildGUI ( );

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

    void HandleMouseEvent(vtkSlicerInteractorStyle *style);

    // Description:
    // Describe behavior at module startup and exit.
    virtual void Enter ( );
    virtual void Exit ( );

    void Init();

    //BTX
    static void DataCallback(vtkObject *caller, 
                unsigned long eid, void *clientData, void *callData);

  //ETX

 protected:
    vtkProstateNavGUI ( );
    virtual ~vtkProstateNavGUI ( );

    //
    // GUI widgets
    //

    // Workphase Frame
    vtkKWPushButtonSet *WorkPhaseButtonSet;


    // Wizard Frame
    vtkKWWizardWidget *WizardWidget;
    vtkProstateNavStep **WizardSteps;

    //Philip Mewes: To show Robots Coordinates and Orientation
    //as a Feedback for the develp core
    vtkKWEntryWithLabel *PositionEntry;
    vtkKWLabel *RobotPositionLabel;

    vtkKWEntryWithLabel *OrientEntry;
    vtkKWLabel *RobotOrientLabel;

    vtkKWEntryWithLabel *NREntry;    
    vtkKWEntryWithLabel *NAEntry;
    vtkKWEntryWithLabel *NSEntry;
    vtkKWEntry          *TREntry;
    vtkKWEntry          *TAEntry;
    vtkKWEntry          *TSEntry;
    vtkKWEntryWithLabel *PREntry;
    vtkKWEntryWithLabel *PAEntry;
    vtkKWEntryWithLabel *PSEntry;
    vtkKWEntryWithLabel *O4Entry;

    vtkKWCheckButton *ConnectCheckButton;
    vtkKWCheckButton *ConnectCheckButtonRI;
    vtkKWCheckButton *ConnectCheckButtonNT;

    vtkKWCheckButton *ConnectCheckButtonStartScanner;
    vtkKWCheckButton *ConnectCheckButtonStopScanner;
    vtkKWCheckButton *ConnectCheckButtonprepScanner;
    vtkKWCheckButton *ConnectCheckButtonpauseScanner;
    vtkKWCheckButton *ConnectCheckButtonresumeScanner;

    vtkKWCheckButton *LocatorCheckButton;
    vtkKWCheckButton *FreezeImageCheckButton;
    vtkKWCheckButton *NeedleCheckButton;

    vtkKWCheckButton *LocatorModeCheckButton;
    vtkKWCheckButton *UserModeCheckButton;

    vtkKWMenuButton *RedSliceMenu;
    vtkKWMenuButton *YellowSliceMenu;
    vtkKWMenuButton *GreenSliceMenu;

    vtkKWLoadSaveButtonWithLabel *LoadConfigButton;
    vtkKWLoadSaveButtonWithLabel *LoadConfigButtonNT;

    vtkKWEntry *ConfigFileEntry;
    vtkKWEntry *ScannerStatusLabelDisp;
    vtkKWEntry *SoftwareStatusLabelDisp;
    vtkKWEntry *RobotStatusLabelDisp;

    vtkKWEntryWithLabel *GetImageSize;
    vtkKWEntryWithLabel *MultiFactorEntry;

    vtkKWPushButton *AddCoordsandOrientTarget;

    vtkKWMultiColumnListWithScrollbars *PointPairMultiColumnList;
    vtkKWMultiColumnListWithScrollbars *TargetListColumnList;

    //    vtkKWPushButton *LoadPointPairPushButton;
    //    vtkKWPushButton *SavePointPairPushButton;
    vtkKWPushButton *DeleteTargetPushButton;
    vtkKWPushButton *DeleteAllTargetPushButton;
    vtkKWPushButton *MoveBWPushButton;
    vtkKWPushButton *MoveFWPushButton;
    vtkKWPushButton *SetOrientButton;

    // Widgets for Calibration Frame
    vtkKWEntry               *CalibImageFileEntry;
    vtkKWPushButton          *ReadCalibImageFileButton;
    vtkKWLoadSaveButtonWithLabel *ListCalibImageFileButton;
    
    // Module logic and mrml pointers
    vtkProstateNavLogic *Logic;

    int SliceDriver0;
    int SliceDriver1;
    int SliceDriver2;

    //Robotcontrollvector
    //BTX
   
    std::vector<float> xsendrobotcoords;
    std::vector<float> ysendrobotcoords;
    std::vector<float> zsendrobotcoords;
    std::vector<float> osendrobotcoords;

    typedef std::vector<float> FloatVector;
    std::vector<FloatVector> sendrobotcoordsvector;
    //ETX

    //BTX
    std::string LocatorModelID;
    std::string LocatorModelID_new;
    //ETX

    vtkIGTDataManager *DataManager;
    vtkIGTPat2ImgRegistration *Pat2ImgReg;
    vtkCallbackCommand *DataCallbackCommand;

    // Access the slice windows
    vtkSlicerSliceLogic *Logic0;
    vtkSlicerSliceLogic *Logic1;
    vtkSlicerSliceLogic *Logic2;
    vtkMRMLSliceNode *SliceNode0;
    vtkMRMLSliceNode *SliceNode1;
    vtkMRMLSliceNode *SliceNode2;
    vtkSlicerSliceControllerWidget *Control0;
    vtkSlicerSliceControllerWidget *Control1;
    vtkSlicerSliceControllerWidget *Control2;

    vtkSlicerVolumesLogic *VolumesLogic;
    //vtkMRMLVolumeNode     *RealtimeVolumeNode;

    int NeedOrientationUpdate0;
    int NeedOrientationUpdate1;
    int NeedOrientationUpdate2;
   
    int RealtimeXsize;
    int RealtimeYsize;
    
    //Workphase State Transition Controll
    
    int NeedRealtimeImageUpdate;

    int RequestedWorkphase;

    void UpdateAll();
    void UpdateLocator(vtkTransform *, vtkTransform *);
    void UpdateSliceDisplay(float nx, float ny, float nz, 
                            float tx, float ty, float tz, 
                            float px, float py, float pz);


 private:
    vtkProstateNavGUI ( const vtkProstateNavGUI& ); // Not implemented.
    void operator = ( const vtkProstateNavGUI& ); //Not implemented.

    // void BuildGUIForHandPieceFrame ();
    void BuildGUIForWorkPhaseFrame();
    void BuildGUIForWizardFrame();
    void BuildGUIForHelpFrame();
    void BuildGUIForTrackingFrame();
    void BuildGUIForDeviceFrame();
    void BuildGUIForRealtimeacqFrame();
    void BuildGUIForscancontrollFrame();
    void BuildGUIForCalibration();

    int  ChangeWorkPhase(int phase, int fChangeWizard=0);

    //void TrackerLoop();


};



#endif
