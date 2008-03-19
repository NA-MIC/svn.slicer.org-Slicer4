/*==========================================================================

Portions (c) Copyright 2008 Brigham and Women's Hospital (BWH) All Rights Reserved.

See Doc/copyright/copyright.txt
or http://www.slicer.org/copyright/copyright.txt for details.

Program:   3D Slicer
Module:    $HeadURL: $
Date:      $Date: $
Version:   $Revision: $

==========================================================================*/

// .NAME vtkOpenIGTLinkLogic - slicer logic class for Locator module 
// .SECTION Description
// This class manages the logic associated with tracking device for
// IGT. 


#ifndef __vtkOpenIGTLinkLogic_h
#define __vtkOpenIGTLinkLogic_h

#include <vector>

#include "vtkOpenIGTLinkWin32Header.h"

#include "vtkSlicerBaseLogic.h"
#include "vtkSlicerModuleLogic.h"
#include "vtkSlicerApplication.h"
#include "vtkCallbackCommand.h"

#include "vtkMRMLFiducialListNode.h"
#include "vtkMRMLSliceNode.h"

#include "vtkMultiThreader.h"

class vtkIGTLConnector;

class VTK_OPENIGTLINK_EXPORT vtkOpenIGTLinkLogic : public vtkSlicerModuleLogic 
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

  enum {
    SLICE_DRIVER_USER    = 0,
    SLICE_DRIVER_LOCATOR = 1,
    SLICE_DRIVER_RTIMAGE = 2
  };
  enum ImageOrient{
    SLICE_RTIMAGE_NONE      = 0,
    SLICE_RTIMAGE_PERP      = 1,
    SLICE_RTIMAGE_INPLANE90 = 2,
    SLICE_RTIMAGE_INPLANE   = 3
  };
  enum {  // Events
    LocatorUpdateEvent      = 50000,
    StatusUpdateEvent       = 50001,
    //SliceUpdateEvent        = 50002,
  };

  //ETX
  
  // Work phase keywords used in NaviTrack (defined in BRPTPRInterface.h)

 public:
  
  static vtkOpenIGTLinkLogic *New();
  
  vtkTypeRevisionMacro(vtkOpenIGTLinkLogic,vtkObject);

  vtkSetMacro ( SelectedConnectorIndex,   int );
  vtkGetMacro ( SelectedConnectorIndex,   int );
  
  vtkSetMacro ( NeedRealtimeImageUpdate0, int );
  vtkGetMacro ( NeedRealtimeImageUpdate0, int );
  vtkSetMacro ( NeedRealtimeImageUpdate1, int );
  vtkGetMacro ( NeedRealtimeImageUpdate1, int );
  vtkSetMacro ( NeedRealtimeImageUpdate2, int );
  vtkGetMacro ( NeedRealtimeImageUpdate2, int );

  vtkSetMacro ( NeedUpdateLocator,       bool );
  vtkGetMacro ( NeedUpdateLocator,       bool );

  vtkSetMacro ( SliceDriver0, int );
  vtkGetMacro ( SliceDriver0, int );
  vtkSetMacro ( SliceDriver1, int );
  vtkGetMacro ( SliceDriver1, int );
  vtkSetMacro ( SliceDriver2, int );
  vtkGetMacro ( SliceDriver2, int );

  vtkGetMacro ( Connection,              bool );

  vtkGetObjectMacro ( LocatorTransform, vtkTransform );
  vtkGetObjectMacro ( LocatorMatrix,    vtkMatrix4x4 );

  void PrintSelf(ostream&, vtkIndent);
  //void AddRealtimeVolumeNode(const char* name);

  //----------------------------------------------------------------
  // Connector Management
  //----------------------------------------------------------------

  void AddConnector();
  void DeleteConnector();
  int  GetNumberOfConnectors();
  vtkIGTLConnector* GetConnector(int id);

 private:
  
  //----------------------------------------------------------------
  // Connector Management
  //----------------------------------------------------------------

  //BTX
  std::vector<vtkIGTLConnector*> ConnectorList;
  //ETX
  int SelectedConnectorIndex;


  //----------------------------------------------------------------
  // Real-time image
  //----------------------------------------------------------------
  
  vtkMRMLVolumeNode     *RealtimeVolumeNode;

  int   NeedRealtimeImageUpdate0;
  int   NeedRealtimeImageUpdate1;
  int   NeedRealtimeImageUpdate2;

  vtkMRMLSliceNode *SliceNode0;
  vtkMRMLSliceNode *SliceNode1;
  vtkMRMLSliceNode *SliceNode2;

  int   SliceDriver0;
  int   SliceDriver1;
  int   SliceDriver2;
  
  bool  ImagingControl;
  bool  NeedUpdateLocator;

  long  RealtimeImageTimeStamp;
  //int   RealtimeImageSerial;
  int   RealtimeImageOrient;


  //----------------------------------------------------------------
  // Locator
  //----------------------------------------------------------------

  // What's a difference between LocatorMatrix and Locator Transform???
  vtkMatrix4x4*         LocatorMatrix;
  vtkTransform*         LocatorTransform;

  bool  Connection;  
  
 protected:
  
  vtkOpenIGTLinkLogic();
  ~vtkOpenIGTLinkLogic();
  vtkOpenIGTLinkLogic(const vtkOpenIGTLinkLogic&);
  void operator=(const vtkOpenIGTLinkLogic&);
  
  static void DataCallback(vtkObject*, unsigned long, void *, void *);

  void UpdateAll();
  void UpdateSliceDisplay();
  void UpdateLocator();

  vtkMRMLVolumeNode* AddVolumeNode(const char*);
  vtkCallbackCommand *DataCallbackCommand;

};

#endif


  
