/*=========================================================================
  
Program:   Visualization Toolkit
Module:    $RCSfile: vtkMiniBirdTrackingSystem.h,v $
Language:  C++
Date:      $Date: 2007/07/12 16:24:29 $
Version:   $Revision: 1.7 $

=========================================================================*/
// .NAME vtkMiniBirdTrackingSystem
// .SECTION Description
// Class for handling communication with the Ascension MiniBird 
// tracking system. 
//=========================================================================*/

#ifndef __vtkMiniBirdTrackingSystem_h
#define __vtkMiniBirdTrackingSystem_h

//#include "vtkLapUSNavSysConfigure.h"
#include "vtkTrackingSystem.h"
#include "Bird.h"

// System baud rate.
#define DEFAULT_BAUD_RATE 115200
// ID for group of tracked sensors.
#define GROUP_ID 0  
// For clarification of system communication
// status.
#define FAILED 0
// Modes of operation.
#define FOB 0
#define STANDALONE 1
#define NOT_SET 2
// Conversion from inches to centimeters.
#define INCH_TO_CM 2.54

class vtkMiniBirdTrackingSystem : public vtkTrackingSystem {
public:  
  // Description: 
  /// Creates a new instance of the class.
  static vtkMiniBirdTrackingSystem *New();
  
  // Description: 
  /// Defines some useful class identity functions.
  vtkTypeRevisionMacro(vtkMiniBirdTrackingSystem, vtkTrackingSystem);
  
  // Description: 
  /// Prints class information.
  void PrintSelf(ostream &os, vtkIndent indent);
  
  // Description: 
  /// Established a connection with a MiniBird tracking 
  /// system. You MUST set the number of sensors, baud rate (or use
  /// the default), and mode of operation BEFORE calling this procedure,
  /// or you will be unable to connect.
  int OpenConnection();
  
  // Description: 
  /// Closes an existing connection with a MiniBird tracking
  /// system.
  int CloseConnection();
  
  // Description: 
  /// Updates stored data for all sensors currently being 
  /// tracked by the system.
  void Poll();
  
  // Description: 
  /// Set the baud rate for the system.
  vtkSetMacro(BaudRate, int);
  
  // Description:
  /// Get the baud rate for the system.
  vtkGetMacro(BaudRate, int);
  
  //Description:
  // Set/Get COM port where we are going to use for connection.
  vtkSetMacro(COMPort, int);
  vtkGetMacro(COMPort, int);
  
  // Description: 
  /// Set the mode of operation.
  vtkSetMacro(ModeOfOperation, int);
  
  // Description:
  /// Get the mode of operation.
  vtkGetMacro(ModeOfOperation, int);
  
  void SetModeOfOperationToFOB() {
    this->SetModeOfOperation(FOB);
  }
  
  void SetModeOfOperationToStandAlone() {
    this->SetModeOfOperation(STANDALONE);
  }
  
  void SetModeOfOperationToNotSet() {
    this->SetModeOfOperation(NOT_SET);
  }
  
protected:
  vtkMiniBirdTrackingSystem();
  ~vtkMiniBirdTrackingSystem();
  
private:
  vtkMiniBirdTrackingSystem(const vtkMiniBirdTrackingSystem&);  // Not implemented.
  void operator=(const vtkMiniBirdTrackingSystem&);  // Not implemented.
  
  // Description: 
  /// Holds system configuration information,
  /// defined in the MiniBird library file.
  BIRDSYSTEMCONFIG SystemConfig;
  
  // Description: 
  /// Holds sensor configuration informaton, 
  /// defined in the MiniBird library file.
  /// This is a pointer to an array of sensor configurations
  /// which is constructed based on the desired number
  /// of tracked sensors.
  BIRDDEVICECONFIG *SensorConfigs;
  
  // Description: 
  /// Mode of operation, either STANDALONE or FOB.
  /// STANDALONE mode means only one sensor is being tracked and
  /// the address of its system is set to 0. FOB mode 
  /// means the addresses begin at 1. The addresses are set via the
  /// dip switches. See the manual for further information.
  int ModeOfOperation;
  
  // Description: 
  /// Data transfer rate.
  int BaudRate;
  int COMPort;
  
};

#endif






