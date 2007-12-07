/*=auto=========================================================================

  Portions (c) Copyright 2007 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: $
  Date:      $Date: $
  Version:   $Revision: $

=========================================================================auto=*/

// .NAME vtkProstateNavLogic - slicer logic class for Locator module 
// .SECTION Description
// This class manages the logic associated with tracking device for
// IGT. 


#ifndef __vtkProstateNavLogic_h
#define __vtkProstateNavLogic_h

#include "vtkProstateNavWin32Header.h"

#include "vtkSlicerBaseLogic.h"
#include "vtkSlicerLogic.h"
#include "vtkSlicerVolumesLogic.h"
#include "vtkSlicerApplication.h"
#include "vtkCallbackCommand.h"

#include "vtkMRMLFiducialListNode.h"


// This will be removed.
#ifndef USE_NAVITRACK
#define USE_NAVITRACK
#endif

#ifdef USE_NAVITRACK
  #include "vtkIGTOpenTrackerStream2.h"
  #include "vtkIGTOpenTrackerStream.h"
  #include "vtkProstateNavDataStream.h"
#endif


class VTK_PROSTATENAV_EXPORT vtkProstateNavLogic : public vtkSlicerLogic 
{

 public:
  //BTX
  enum WorkPhase {
    StartUp = 0,
    Planning,
    Calibration,
    Targeting,
    Manual,
    Emergency,
    NumPhases,
  };
  enum ImageOrient{
    SLICE_RTIMAGE_PERP      = 0,
    SLICE_RTIMAGE_INPLANE90 = 1,
    SLICE_RTIMAGE_INPLANE   = 2
  };
  enum {  // Events
    LocatorUpdateEvent      = 50000,
    StatusUpdateEvent       = 50001,
    SliceUpdateEvent        = 50002,
  };

  //ETX
  
  // Work phase keywords used in NaviTrack (defined in BRPTPRInterface.h)
  static const char* WorkPhaseKey[vtkProstateNavLogic::NumPhases];

 public:
  
  static vtkProstateNavLogic *New();
  
  vtkTypeRevisionMacro(vtkProstateNavLogic,vtkObject);
  
  vtkGetMacro ( CurrentPhase,         int );
  vtkGetMacro ( PrevPhase,            int );
  vtkGetMacro ( PhaseTransitionCheck, bool );
  vtkSetMacro ( PhaseTransitionCheck, bool );
  vtkGetMacro ( RealtimeImageOrient,  int  );
  vtkSetMacro ( RealtimeImageOrient,  int  );
  
  vtkSetMacro ( NeedRealtimeImageUpdate0, int );
  vtkGetMacro ( NeedRealtimeImageUpdate0, int );
  vtkSetMacro ( NeedRealtimeImageUpdate1, int );
  vtkGetMacro ( NeedRealtimeImageUpdate1, int );
  vtkSetMacro ( NeedRealtimeImageUpdate2, int );
  vtkGetMacro ( NeedRealtimeImageUpdate2, int );
  vtkSetMacro ( ImagingControl,          bool );
  vtkGetMacro ( ImagingControl,          bool );
  vtkSetMacro ( UpdateLocator,           bool );
  vtkGetMacro ( UpdateLocator,           bool );

  vtkGetMacro ( Connection,              bool );
  vtkGetMacro ( RobotWorkPhase,           int );
  vtkGetMacro ( ScannerWorkPhase,         int );

  vtkGetObjectMacro ( LocatorTransform, vtkTransform );
  vtkGetObjectMacro ( LocatorMatrix,    vtkMatrix4x4 );


  void PrintSelf(ostream&, vtkIndent);
  
  void AddRealtimeVolumeNode(vtkSlicerApplication* app, const char* name);

  //void AddRealtimeVolumeNode(vtkSlicerVolumesLogic*, const char*);
  int  SwitchWorkPhase(int);
  int  IsPhaseTransitable(int);
  
  int  ConnectTracker(const char* filename);
  int  DisconnectTracker();
  
  int  RobotStop();
  int  RobotMoveTo(float px, float py, float pz,
                   float nx, float ny, float nz,
                   float tx, float ty, float tz);
  int  RobotMoveTo(float position[3], float orientation[4]);
  
  int  ScanStart();
  int  ScanPause();
  int  ScanStop();
  
  //BTX
  Image* ReadCalibrationImage(const char* filename, int* width, int* height,
                              std::vector<float>& position, std::vector<float>& orientation);
  //ETX
  
  int WorkPhaseStringToID(const char* string);


 private:
  
  static const int PhaseTransitionMatrix[NumPhases][NumPhases];
  
  int   CurrentPhase;
  int   PrevPhase;
  int   PhaseComplete;
  bool  Connected;
  bool  PhaseTransitionCheck;
  bool  RealtimeImageUpdate;
  
  int   NeedRealtimeImageUpdate0;
  int   NeedRealtimeImageUpdate1;
  int   NeedRealtimeImageUpdate2;
  
  bool  ImagingControl;
  bool  UpdateLocator;
  
  int   RealtimeImageSerial;
  int   RealtimeImageOrient;

  // Junichi Tokuda on 11/27/2007:
  // What's a difference between LocatorMatrix and Locator Transform???
  vtkMatrix4x4*         LocatorMatrix;
  vtkTransform*         LocatorTransform;

  vtkMRMLVolumeNode     *RealtimeVolumeNode;
  vtkSlicerVolumesLogic *VolumesLogic;

  bool  Connection;  
  int   RobotWorkPhase;
  int   ScannerWorkPhase;
  
#ifdef USE_NAVITRACK
  vtkIGTOpenTrackerStream2 *OpenTrackerStream2;
  //vtkIGTOpenTrackerStream *OpenTrackerStream;
  vtkProstateNavDataStream *OpenTrackerStream;
#endif

 protected:
  
  vtkProstateNavLogic();
  ~vtkProstateNavLogic();
  vtkProstateNavLogic(const vtkProstateNavLogic&);
  void operator=(const vtkProstateNavLogic&);
  
  static void DataCallback(vtkObject*, unsigned long, void *, void *);
  void UpdateAll();
  vtkMRMLVolumeNode* AddVolumeNode(vtkSlicerVolumesLogic*, const char*);
  
  vtkCallbackCommand *DataCallbackCommand;

};

#endif


  
