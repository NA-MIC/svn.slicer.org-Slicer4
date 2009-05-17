/*=========================================================================

Program:   Visualization Toolkit
Module:    $RCSfile: vtkMicroBirdTrackingSystem.h,v $
Language:  C++
Date:      $Date: 2008/05/27 16:15:34 $
Version:   $Revision: 1.5 $

=========================================================================*/
// .NAME vtkMicroBirdTrackingSystem
// .SECTION Description
// Class for handling communication with the Ascension MicroBird 
// tracking system. 
//========================================================================

#ifndef __vtkMicroBirdTrackingSystem_h
#define __vtkMicroBirdTrackingSystem_h

//#include "vtkLapUSNavSysConfigure.h"
#include "vtkTrackingSystem.h"
#include "PCIBird3.h"

class vtkMicroBirdTrackingSystem : public vtkTrackingSystem {
public:  
  
  static vtkMicroBirdTrackingSystem *New();
  vtkTypeRevisionMacro(vtkMicroBirdTrackingSystem, vtkTrackingSystem);
  void PrintSelf(ostream &os, vtkIndent indent);
  
  int OpenConnection();
  int CloseConnection();
  void Poll();
  
  vtkGetMacro(NumAttachedSensors, int);
  vtkGetMacro(NumAttachedTransmitters, int);
  
  //Description:
  // Set/Get Measurement rate
  vtkSetMacro(MeasurementRate, double);
  vtkGetMacro(MeasurementRate, double);
  
  
  
protected:
  vtkMicroBirdTrackingSystem();
  ~vtkMicroBirdTrackingSystem();
  
private:
  vtkMicroBirdTrackingSystem(const vtkMicroBirdTrackingSystem&);  // Not implemented.
  void operator=(const vtkMicroBirdTrackingSystem&);  // Not implemented.
  
  int NumAttachedSensors;
  int NumAttachedTransmitters;
  
  SYSTEM_CONFIGURATION SystemConfig;
  SENSOR_CONFIGURATION *SensorConfigs;
  TRANSMITTER_CONFIGURATION *TransmitterConfigs;
  
  // Description:
  // Tracking system variables
  int MeasurementRate;
};

#endif






