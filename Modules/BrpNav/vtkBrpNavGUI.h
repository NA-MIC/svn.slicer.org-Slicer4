// .NAME vtkBrpNavGUI 
// .SECTION Description
// Main Volumes GUI and mediator methods for slicer3. 


#ifndef __vtkBrpNavGUI_h
#define __vtkBrpNavGUI_h


#ifdef WIN32
#include "vtkBrpNavWin32Header.h"
#endif

#include "vtkSlicerModuleGUI.h"
#include "vtkBrpNavLogic.h"

#include "vtkIGTDataManager.h"
#include "vtkIGTPat2ImgRegistration.h"
#include "vtkCallbackCommand.h"
#include "vtkSlicerInteractorStyle.h"
#include "vtkSlicerVolumesLogic.h"


#include <string>

#ifdef USE_OPENTRACKER
#include "vtkIGTOpenTrackerStream.h"

#define PREP_PHASE = 1;
#define PLANNING_PHASE = 2;
#define CALIB_PHASE = 3;
#define TARG_PHASE = 4;
#define MANU_PHASE = 5;
#define EMER_PHASE = 6;

#endif
#ifdef USE_IGSTK
#include "vtkIGTIGSTKStream.h"
#endif

class vtkKWPushButton;
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


// Description:    
// This class implements Slicer's Volumes GUI
//
class VTK_BRPNAV_EXPORT vtkBrpNavGUI : public vtkSlicerModuleGUI
{
 public:
    // Description:    
    // Usual vtk class functions
    static vtkBrpNavGUI* New (  );
    vtkTypeRevisionMacro ( vtkBrpNavGUI, vtkSlicerModuleGUI );
    void PrintSelf (ostream& os, vtkIndent indent );
   

    
    //SendDATANavitrack
    // Description:    
    // Get methods on class members (no Set methods required)
    vtkGetObjectMacro ( Logic, vtkBrpNavLogic );

    // Description:
    // API for setting VolumeNode, VolumeLogic and
    // for both setting and observing them.
    void SetModuleLogic ( vtkBrpNavLogic *logic )
        { this->SetLogic ( vtkObjectPointer (&this->Logic), logic ); }
    void SetAndObserveModuleLogic ( vtkBrpNavLogic *logic )
        { this->SetAndObserveLogic ( vtkObjectPointer (&this->Logic), logic ); }

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
    vtkBrpNavGUI ( );
    virtual ~vtkBrpNavGUI ( );
    vtkKWEntryWithLabel *NormalOffsetEntry;
    vtkKWEntryWithLabel *TransOffsetEntry;
    vtkKWEntryWithLabel *NXTOffsetEntry;

    vtkKWEntryWithLabel *NormalSizeEntry;
    vtkKWEntryWithLabel *TransSizeEntry;
    vtkKWEntryWithLabel *RadiusEntry;

    vtkKWEntryWithLabel *setSpeedEntry;

    vtkKWEntryWithLabel *NREntry;    
    vtkKWEntryWithLabel *NAEntry;
    vtkKWEntryWithLabel *NSEntry;
    vtkKWEntry *TREntry;
    vtkKWEntry *TAEntry;
    vtkKWEntry *TSEntry;
    vtkKWEntryWithLabel *PREntry;
    vtkKWEntryWithLabel *PAEntry;
    vtkKWEntryWithLabel *PSEntry;
    vtkKWEntryWithLabel *O4Entry;

    vtkKWFrame *ExtraFrame;

    /*
    vtkKWScaleWithEntry* RedColorScale;
    vtkKWScaleWithEntry* GreenColorScale;
    vtkKWScaleWithEntry* BlueColorScale;    
    */
    


    vtkKWMenuButtonWithLabel *ServerMenu;
    vtkKWMenuButtonWithLabel *PauseCheckButton;
    vtkKWMenuButtonWithLabel *SetText;
   
  
    vtkKWCheckButton *ConnectCheckButton;
    vtkKWCheckButton *ConnectCheckButtonRI;
    vtkKWCheckButton *ConnectCheckButtonNT;

    vtkKWCheckButton *ConnectCheckButtonSEND;
    vtkKWCheckButton *ConnectCheckButtonPASSROBOTCOORDS;
    vtkKWCheckButton *ConnectCheckButtonStartScanner;
    vtkKWCheckButton *ConnectCheckButtonStopScanner;
    vtkKWCheckButton *ConnectCheckButtonprepScanner;
    vtkKWCheckButton *ConnectCheckButtonpauseScanner;
    vtkKWCheckButton *ConnectCheckButtonresumeScanner;
   



    vtkKWCheckButton *ConnectCheckButtonnewexam;
    vtkKWCheckButton *ConnectCheckButtonsetprotocol;

    vtkKWCheckButton *LocatorCheckButton;
    vtkKWCheckButton *FreezeImageCheckButton;
    vtkKWCheckButton *NeedleCheckButton;
    
    vtkKWCheckButton *WorkPhaseStartUpButton;
    vtkKWCheckButton *WorkPhasePlanningButton;
    vtkKWCheckButton *WorkPhaseCalibarationButton;
    vtkKWCheckButton *WorkPhaseTargetingButton;
    vtkKWCheckButton *WorkPhaseManualButton;
    vtkKWCheckButton *WorkPhaseEmergencyButton;
    


    vtkKWCheckButton *HandleCheckButton;
    vtkKWCheckButton *GuideCheckButton;

    vtkKWCheckButton *LocatorModeCheckButton;
    vtkKWCheckButton *UserModeCheckButton;

    vtkKWMenuButton *RedSliceMenu;
    vtkKWMenuButton *YellowSliceMenu;
    vtkKWMenuButton *GreenSliceMenu;

    //#ifdef USE_OPENTRACKER
    vtkKWLoadSaveButtonWithLabel *LoadConfigButton;
    vtkKWLoadSaveButtonWithLabel *LoadConfigButton2;
    vtkKWLoadSaveButtonWithLabel *LoadConfigButtonNT;
    vtkKWLoadSaveButtonWithLabel *LoadConfigButtonRI;
    vtkKWEntry *ConfigFileEntry;
    vtkKWEntry *ScannerStatusLabelDisp;
    vtkKWEntry *SoftwareStatusLabelDisp;
    vtkKWEntry *RobotStatusLabelDisp;
    vtkKWEntry *ConfigFileEntry2;
    vtkKWEntry *ConfigFileEntryRI;


      vtkKWEntryWithLabel *positionbrpy;
    vtkKWEntryWithLabel *positionbrpz;
    vtkKWEntryWithLabel *positionbrpx;
    
    vtkKWEntryWithLabel *positionbrppatientweight;
    vtkKWEntryWithLabel *positionbrppatientid;
    vtkKWEntryWithLabel *positionbrppatientname;
    vtkKWEntryWithLabel *positionbrpsetprotocol;

    vtkKWEntryWithLabel *orientationbrpo1;
    vtkKWEntryWithLabel *orientationbrpo2;
    vtkKWEntryWithLabel *orientationbrpo3;
    vtkKWEntryWithLabel *orientationbrpo4;


    //#endif
#ifdef USE_IGSTK
    vtkKWMenuButtonWithLabel *DeviceMenuButton;
    vtkKWMenuButtonWithLabel *PortNumberMenuButton;
    vtkKWMenuButtonWithLabel *BaudRateMenuButton;
    vtkKWMenuButtonWithLabel *DataBitsMenuButton;
    vtkKWMenuButtonWithLabel *ParityTypeMenuButton;
    vtkKWMenuButtonWithLabel *StopBitsMenuButton;
    vtkKWMenuButtonWithLabel *HandShakeMenuButton;
#endif

    vtkKWEntryWithLabel *UpdateRateEntry;
    vtkKWEntryWithLabel *GetImageSize;
    vtkKWEntryWithLabel * MultiFactorEntry;

    vtkKWEntryWithLabel *PatCoordinatesEntry;
    vtkKWEntryWithLabel *SlicerCoordinatesEntry;
    vtkKWPushButton *GetPatCoordinatesPushButton;
    vtkKWPushButton *AddPointPairPushButton;
    vtkKWPushButton *AddCoordsandOrientTarget;

    vtkKWMultiColumnListWithScrollbars *PointPairMultiColumnList;
    vtkKWMultiColumnListWithScrollbars *TargetListColumnList;

    //    vtkKWPushButton *LoadPointPairPushButton;
//    vtkKWPushButton *SavePointPairPushButton;
    vtkKWPushButton *DeletePointPairPushButton;
    vtkKWPushButton *DeleteTargetPushButton;
    vtkKWPushButton *DeleteAllTargetPushButton;
    vtkKWPushButton *DeleteAllPointPairPushButton;
    vtkKWPushButton *MoveBWPushButton;
    vtkKWPushButton *MoveFWPushButton;
    vtkKWPushButton *SetOrientButton;
    vtkKWPushButton *RegisterPushButton;
    vtkKWPushButton *ResetPushButton;


    // Module logic and mrml pointers
    vtkBrpNavLogic *Logic;

    // int StopTimer;
    vtkMatrix4x4 *LocatorMatrix;
    vtkMatrix4x4 *LocatorMatrix_cb2;
      
    //Robotcontrollvector
    //BTX

    typedef std::vector<float> FloatVector;
   
    std::vector<float> xsendrobotcoords;
    std::vector<float> ysendrobotcoords;
    std::vector<float> zsendrobotcoords;
    
    std::vector<float> osendrobotcoords;

    std::vector<FloatVector> sendrobotcoordsvector;
      //ETX
    //RI


    vtkMRMLModelNode *LocatorModelNode;
    vtkMRMLModelDisplayNode *LocatorModelDisplayNode;

    //BTX
    std::string LocatorModelID;
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
    vtkMRMLVolumeNode     *RealtimeVolumeNode;

  

    int NeedOrientationUpdate0;
    int NeedOrientationUpdate1;
    int NeedOrientationUpdate2;
   
     int RealtimeXsize;
     int RealtimeYsize;
     
     // int RealtimeImageData;
      float  xsizevalueRI;
     float  ysizevalueRI;

    
    

    int RealtimeImageSerial;
    int NeedRealtimeImageUpdate;

    //status check
    int var_status_scanner;
    int var_status_soft;
    int var_status_robot;
    
    

    char xmlpathfilename[256];
    char xcoordsrobot[12];
    char ycoordsrobot[12];
    char zcoordsrobot[12];

    char o1coordsrobot[12];
    char o2coordsrobot[12];
    char o3coordsrobot[12];
    char o4coordsrobot[12];

    float brptmp;
    void UpdateAll();
    void UpdateLocator();
    void UpdateRealtimeImg();
    void UpdateSliceDisplay(float nx, float ny, float nz, 
                            float tx, float ty, float tz, 
                            float px, float py, float pz);

    void brpxml(const char* xmlpathfilename);

 private:
    vtkBrpNavGUI ( const vtkBrpNavGUI& ); // Not implemented.
    void operator = ( const vtkBrpNavGUI& ); //Not implemented.

    // void BuildGUIForHandPieceFrame ();
    void BuildGUIForTrackingFrame ();
   
    void BuildGUIForDeviceFrame (); 
     void BuildGUIForRealtimeacqFrame ();
    
    void BuildGUIForWorkPhaseFrame ();
    void BuildGUIForscancontrollFrame ();
    void SetOpenTrackerforBRPDataFlowValveFilter ();
    
    void TrackerLoop();


#ifdef USE_OPENTRACKER
    vtkIGTOpenTrackerStream *OpenTrackerStream;
    void SetOpenTrackerConnectionParameters();
 
    void SetOpenTrackerConnectionCoordandOrient();
    void SetOpenTrackerforScannerControll();
    void GetSizeforRealtimeImaging();
    void GetImageDataforRealtimeImaging();
    void SetOrientationforRobot();
    void GetCoordsOrientforScanner();
    Image  ImageDataRI;

#endif

#ifdef USE_IGSTK    
    vtkIGTIGSTKStream *IGSTKStream;
    void SetIGSTKConnectionParameters();
#endif


vtkMRMLVolumeNode* AddVolumeNode(vtkSlicerVolumesLogic*, const char*);


};



#endif
