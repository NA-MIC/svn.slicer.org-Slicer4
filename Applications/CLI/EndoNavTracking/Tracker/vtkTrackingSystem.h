/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkTrackingSystem.h,v $
  Language:  C++
  Date:      $Date: 2007/12/18 06:05:16 $
  Version:   $Revision: 1.15 $

=========================================================================*/
// .NAME vtkTrackingSystem
// .SECTION Description
// Abstract class for communicating with sensor tracking systems. 
//=========================================================================*/

#ifndef __vtkTrackingSystem_h
#define __vtkTrackingSystem_h

//#include "vtkLapUSNavSysConfigure.h"
#include "vtkSystemIncludes.h"
#include "vtkMatrix4x4.h"
#include "vtkTransform.h"
#include "vtkLinearTransform.h"
#include "vtkDoubleArray.h"
#include "vtkCharArray.h"
#include "vtkProcessObject.h"
#include "vtkObjectFactory.h"
#include <fstream>
#include <string>
#include <vector>
using namespace std;

// Explicit true and false values.
#define TRUE 1
#define FALSE 0
// Conversions.
#define POSK36 (float)(36.0/32768.0) // integer to inches 
#define WTF (float)(1.0/32768.0) // float to word integer
// Pi.
#define PI 3.1415926535897932384626433832795028841971693993751



class  vtkTrackingSystem : public vtkProcessObject {
public:

  static vtkTrackingSystem *New();
  vtkTypeRevisionMacro(vtkTrackingSystem, vtkProcessObject);
  void PrintSelf(ostream &os, vtkIndent indent);

  virtual int    OpenConnection();
  virtual int    CloseConnection();
  virtual void  Poll();
  void      UpdateLocatorMatrices();

  void      SetCalibrationMatrix(int index, vtkMatrix4x4 *matrix);
  void      SetCalibrationMatrixElement(int index, int i, int j, double value);
  vtkMatrix4x4  *GetCalibrationMatrix(int index);
  double      GetCalibrationMatrixElement(int index, int i, int j);
    
  void      SetRegistrationTransform(vtkLinearTransform *transform);
  void      SetRegistrationMatrix(vtkMatrix4x4 *matrix);
  void      SetRegistrationMatrixElement(int i, int j, double value);
  vtkMatrix4x4  *GetRegistrationMatrix();
  double      GetRegistrationMatrixElement(int i, int j);

  void      SetSensorMatrix(int index, vtkMatrix4x4 *matrix);
  vtkMatrix4x4  *GetSensorMatrix(int index);
  vtkDoubleArray  *GetSensorEulerAngles(int index);
  
  vtkMatrix4x4  *GetLocatorMatrix(int index);

  double      GetSensorTimeStamp(int index);

  char  *GetXAxisLabel(int sensor);
  char  *GetYAxisLabel(int sensor);
        char  *GetZAxisLabel(int sensor);

  void      BuildMatrixFromEulerAngles(vtkDoubleArray **all_data, int index, vtkMatrix4x4 *matrix); 
  void      BuildMatrixFromEulerAngles(vtkDoubleArray *angles, vtkMatrix4x4 *matrix); 
  void      BuildMatrixFromEulerAngles(double x, double y, double z, double alpha, double beta, double gamma, double mat[4][4]);
  void      BuildEulerAnglesFromMatrix(vtkMatrix4x4 * matrix, vtkDoubleArray *angles);
  void      BuildEulerAnglesFromMatrix(double mat[4][4], double &x, double &y, double &z, double &alpha, double &beta, double &gamma);
    void      SetEulerAnglesInDegrees() {this->EulerAnglesInDegrees = 1; this->Modified();};
  void      SetEulerAnglesInRadians() {this->EulerAnglesInDegrees = 0; this->Modified();};      

  vtkGetMacro(IsConnected, int);
  vtkSetMacro(NumSensors, int);
  vtkGetMacro(NumSensors, int);
  vtkSetMacro(NumTransmitters, int);
  vtkGetMacro(NumTransmitters, int);


protected:
  vtkTrackingSystem();
  ~vtkTrackingSystem();
  
  int        IsConnected;
  int        NumSensors;  
  int        NumTransmitters;

  vtkDoubleArray  **SensorEulerAngles; 
  vtkMatrix4x4  **SensorMatrices;
  vtkMatrix4x4  **CalibrationMatrices;
  vtkMatrix4x4  **LocatorMatrices;
  vtkMatrix4x4  *RegistrationMatrix;
  vtkLinearTransform *RegistrationTransform;

  double *SensorTimeStamps;

  int EulerAnglesInDegrees;

    // Containers for labels
//BTX
  std::vector<std::string> XAxisLabel;
  std::vector<std::string> YAxisLabel;
  std::vector<std::string> ZAxisLabel;
  void LabelPlaneAxes();
  void FindMaxAxis(std::string &str,double x[4]);
//ETX
        
  char *activeXAxisLabel;
  char *activeYAxisLabel;
        char *activeZAxisLabel;
  
  // Description: 
  /// Log for tracking system activity.
  char *LogFile;
  fstream Log;
  
private:
  vtkTrackingSystem(const vtkTrackingSystem&);  // Not implemented.
  void operator=(const vtkTrackingSystem&);     // Not implemented.

};

#endif












