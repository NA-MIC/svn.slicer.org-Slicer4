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

#include "vtkCallbackCommand.h"

#include "vtkMRMLFiducialListNode.h"


// This will be removed.
#ifndef USE_NAVITRACK
#define USE_NAVITRACK
#endif

#ifdef USE_NAVITRACK
  #include "vtkIGTOpenTrackerStream.h"
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
  //ETX
  
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
  
  void PrintSelf(ostream&, vtkIndent);
  
  void AddRealtimeVolumeNode(vtkSlicerVolumesLogic*, const char*);
  int  SwitchWorkPhase(int);
  int  IsPhaseTransitable(int);
  
  int  ConnectTracker(const char* filename);
  int  DisconnectTracker();
  
  int  RobotStop();
  int  RobotMoveTo(float px, float py, float pz,
                   float nx, float ny, float nz,
                   float tx, float ty, float tz);
  
  int  ScanStart();
  int  ScanPause();
  int  ScanStop();
  
  //BTX
  Image* ReadCalibrationImage(const char* filename, int* width, int* height,
                              std::vector<float>& position, std::vector<float>& orientation);
  //ETX
  
 private:
  
  static const int PhaseTransitionMatrix[NumPhases][NumPhases];
  
  int   CurrentPhase;
  int   PrevPhase;
  int   PhaseComplete;
  bool  Connected;
  bool  PhaseTransitionCheck;
  bool  OrientationUpdate;
  
  int   NeedRealtimeImageUpdate0;
  int   NeedRealtimeImageUpdate1;
  int   NeedRealtimeImageUpdate2;
  
  bool  ImagingControl;
  bool  UpdateLocator;
  
  int   RealtimeImageSerial;
  int   RealtimeImageOrient;
  
  vtkMatrix4x4          *LocatorMatrix;
  vtkMRMLVolumeNode     *RealtimeVolumeNode;
  vtkSlicerVolumesLogic *VolumesLogic;
  
  
#ifdef USE_NAVITRACK
  vtkIGTOpenTrackerStream *OpenTrackerStream;
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


  
