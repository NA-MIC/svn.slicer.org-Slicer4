/*=========================================================================

Program:   Visualization Toolkit
Module:    $RCSfile: vtkPCIBirdWin32TrackedVideoSource.h,v $
Language:  C++
Date:      $Date: 2007/07/12 16:24:29 $
Version:   $1.0$

=========================================================================*/
// .NAME vtkPCIBirdWin32TrackedVideoSource - Video-for-Windows video digitizer
// with Flock of Birds instrument tracking.
// .SECTION Description
// vtkPCIBirdWin32TrackedVideoSource grabs frames or streaming video from a
// Video for Windows compatible device on the Win32 platform.  It also
// captures tracking data from a flock of birds system.


#ifndef __vtkPCIBirdWin32TrackedVideoSource_h
#define __vtkPCIBirdWin32TrackedVideoSource_h

#define MAX_SENSORS 2
#define NUM_TOOLS 2
#define INCH_TO_CM 25.4
#define GROUP_ID  0
#define NUM_PORTS 2
#define POSK36 (float)(36.0/32768.0) 
#define WTF (float)(1.0/32768.0)     
#define PI 3.14159265358979323846

#include <vtkLapUSNavSysConfigure.h>
#include <vtkWin32VideoSource.h>
#include <vtkObjectFactory.h>
#include <vtkMatrix4x4.h>
#include <vtkTransform.h>
#include <vtkImageData.h>
#include <vtkFloatArray.h>
#include <vtkDoubleArray.h>
#include <windows.h>
#include <winuser.h>
#include <vfw.h>
#include <time.h>
#include <stdio.h>
#include <process.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
#include "PCIBird3.h"

using namespace std;

class VTK_LAPUSNAVSYS_EXPORT vtkPCIBirdWin32TrackedVideoSource : public vtkWin32VideoSource
{
public:  
  static vtkPCIBirdWin32TrackedVideoSource *New();
  vtkTypeMacro(vtkPCIBirdWin32TrackedVideoSource,vtkWin32VideoSource);
  void PrintSelf(ostream& os, vtkIndent indent);
  
  // --------------------

  /** LOCALIZER COMMUNICATION PROCEDURES **/

  // Establishes a connection with the PCI flock of birds device. It first obtains
  // the system configuration, then the configurations of individual sensor, and
  // then transmitter configurations. It then establishes a connection with the first
  // transmitter it finds. Right now, none of the arguments being passed are utilized
  // within the procedure.
  int OpenConnection(int port, int baud, int numbirds, int video);

  // Updates the system information once a connection has been established. This information
  // includes the sensor matrix data for each sensor and any video data that may be associated
  // with a device one of the sensors is connected to, such as the ultrasound images sent from
  // a laparoscopic ultrasound probe.
  void Poll();

  // Not currently being utilized, but its normal functionality would be to close the 
  // connection with the PCI flock of birds device.
  void CloseConnection();

  // Locator matrices are maintained for each sensor, and these matrices provide the position and
  // orientation data for the sensors in the Slicer frame. In other words, they are the original
  // sensor matrices after the calibration and registration matrices have been applied to them.
  void UpdateLocatorMatrix(int numPort);
    
  // Record video and position in a buffer (VideoBuffer and SensorBuffer). This method is used as a recording
  // tool for calibration and volume scannings.
  void Record();
   
  // Stop video and position recording
  void Stop();
 
  // --------------------

  /** TCL COMMUNICATION PROCEDURES **/

  // set...

  // Sets an individual element of the specified calibration matrix (indicated by tool ID) to
  // the specified value.
  void SetCalibrationMatrixElement(int idtool, int i, int j, double entry);

  // Sets the specified calibration matrix to the matrix passed to the procedure.
  void SetCalibrationMatrix(int idtool, vtkMatrix4x4 *calib_matrix);

  // Sets an individual element of the registration matrix to the specified value.
  void SetRegistrationMatrixElement(int i, int j, double entry);

  // Sets the registration matrix to the matrix passed to the procedure.
  void SetRegistrationMatrix(vtkMatrix4x4 *reg_matrix);
  
  // Returns an individual element of the specified calibration matrix.
  double GetCalibrationMatrixElement(int idtool, int i, int j);

  // Returns the specified calibration matrix.
  vtkMatrix4x4 *GetCalibrationMatrix(int idtool);

  // Returns an individual element of the registration matrix.
  double GetRegistrationMatrixElement(int i, int j);

  // Returns the registration matrix.
  vtkMatrix4x4 *GetRegistrationMatrix();
  
  // Returns the bird matrix for the sensor on the specified port.
  vtkMatrix4x4 *GetSensorMatrix(int numPort);

  // Returns the locator matrix for the sensor on the specified port. 
  vtkMatrix4x4 *GetLocatorMatrix(int numPort);
  
  // Returns the orientation data for the sensor attached to the 
  // laparoscopic probe. This information is used for model
  // reconstruction.
  vtkDoubleArray *GetProbeBirdAngle(int numPort); 
    
  // Returns the time of acquisition for the sensor on the specified port.
  int GetSensorTime(int numPort);

  // Get/Set Method for maximum number of frames that are recorded when
  // the record method is invoked.
  vtkGetMacro(MaxNumberOfFrames,int);
  vtkSetMacro(MaxNumberOfFrames,int);

  // Get/Set the port that we use to record positions that will be stored in SensorBuffer.
  vtkGetMacro(RecordingPort,int);
  vtkSetMacro(RecordingPort,int);
  
  vtkGetMacro(StopFlag,int);
  
  vtkGetMacro(Doppler,int);
  vtkSetMacro(Doppler,int);
  vtkBooleanMacro(Doppler,int);

  // Access to the recorded buffers: video as a vtkImageData and sensor positions as a 
  // vtkFloatArray with 6 components: x,y,z,alpha,beta,gamma.
  vtkGetObjectMacro(VideoBuffer,vtkImageData);
  vtkGetObjectMacro(SensorBuffer,vtkDoubleArray);

  // Methods for conversion between matrix form 4x4 and Euler angles
  void BuildMatrixFromEulerAngles(vtkDoubleArray * angles, vtkMatrix4x4 *matrix);
  void BuildMatrixFromEulerAngles(double x,double y,double z,double alpha,
                                  double beta,double gamma,double mat[4][4]);
  void BuildEulerAnglesFromMatrix(vtkMatrix4x4 * matrix, vtkDoubleArray *angles);
  void BuildEulerAnglesFromMatrix(double mat[4][4],double& x, double& y, 
                                  double&z, double &alpha, double &beta, double &gamma);

protected:
  vtkPCIBirdWin32TrackedVideoSource();
  ~vtkPCIBirdWin32TrackedVideoSource();
  vtkPCIBirdWin32TrackedVideoSource(const vtkPCIBirdWin32TrackedVideoSource&) {};
  void operator=(const vtkPCIBirdWin32TrackedVideoSource&) {};
  
  // not currently in use 
  int m_nSensors; // number of sensors
  int m_ActiveSensorsID[16];   // table of sensor ID's
  bool bStandalone; 
  int NumLocators; 
  int m_BirdInitialized; 
  int ActiveLocator; 
  int useVideo; 
  int time;
  
  // Additional flag when recording. Given that we rely on Grab to record 
  // a stream Recording must be set to 0. We need an additional variable to control when
  // we are recording.
  int RecordingUS;
  int StopFlag;
  int Doppler;
  
  // stores calibration matrices for the tools (palpator/stylus and lapus probe)
  vtkMatrix4x4 *m_CalibrationMatrix[NUM_TOOLS];

  // stores position and orientation data for each
  // sensor within transmitter frame; in other words, the
  // raw sensor data
  vtkMatrix4x4 *m_SensorMatrixInLoc[MAX_SENSORS];

  // stores registration matrix which converts position/orientation
  // information from transmitter to Slicer frame
  vtkMatrix4x4 *m_RegistrationMatrix;

  // stores Slicer position and orientation data for
  // each active sensor, which is the raw data after
  // being transformed using calibration and registration
  // matrices
  vtkMatrix4x4 *m_LocatorMatrix[NUM_TOOLS] ;

  // stores orientation angles for the probe sensor
  vtkDoubleArray *m_ProbeBirdAngles[MAX_SENSORS];

  // stores the time 
  int m_SensorTime[MAX_SENSORS];

  int RecordingPort;

  // MAGNETIC LOCALIZER CONFIGURATION : these variables set/get configuration 
  // parameters for all parts (PCI card, sensor(s), transmitter) of the PCI flock of 
  // birds system 

  SYSTEM_CONFIGURATION    m_SystemConfig ;
  SENSOR_CONFIGURATION    m_SensorConfig[2] ;
  TRANSMITTER_CONFIGURATION  m_TransmitterConfig[1];

  // Recording buffers
  vtkDoubleArray *SensorBuffer;
  vtkImageData *VideoBuffer;
  int MaxNumberOfFrames;

  FILE*  m_LogFile;
  FILE* m_DataFile;

  //void  ErrorMessage(int pErrorCode);

};

#endif
