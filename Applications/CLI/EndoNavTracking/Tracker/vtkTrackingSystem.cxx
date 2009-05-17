/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkTrackingSystem.cxx,v $
  Language:  C++
  Date:      $Date: 2008/06/02 15:53:10 $
  Version:   $Revision: 1.15 $

=========================================================================*/

#include "vtkTrackingSystem.h"

vtkCxxRevisionMacro(vtkTrackingSystem, "$Revision: 1.15 $");
vtkStandardNewMacro(vtkTrackingSystem);

vtkTrackingSystem::vtkTrackingSystem() {

  // Set connection status.
  this->IsConnected = FALSE;

  // Set number of sensors.
  this->NumSensors = 0;

  // Set number of transmitters.
  this->NumTransmitters = 0;

  // Set number of components for sensor data.
  this->SensorEulerAngles = NULL;

  // Initialize sensor matrices.
  this->SensorMatrices = NULL;

  // Initialize calibration matrices.
  this->CalibrationMatrices = NULL;

  // Initialize registration matrix.
  this->RegistrationTransform = NULL;
  this->RegistrationMatrix = NULL;

  // Initialize locator matrices.
  this->LocatorMatrices = NULL;

  // Initialzie time stamps.
  this->SensorTimeStamps = NULL;
  
  //EulerAngles in degrees
  this->EulerAnglesInDegrees = 0;
        
  this->activeXAxisLabel = new char[4];
  this->activeYAxisLabel = new char[4];
        this->activeZAxisLabel = new char[4];

  // Establish log file.
  this->LogFile = "tracking_log.txt";

  // Open log file.
  this->Log.open(this->LogFile, fstream::out);

}

vtkTrackingSystem::~vtkTrackingSystem() {

  // Close the connection if it is has
  // not been.
  if (this->IsConnected == TRUE) 
    this->CloseConnection();
  
  if (this->RegistrationTransform)
     this->RegistrationTransform->Delete();
  if (this->RegistrationMatrix)
     this->RegistrationMatrix->Delete();   
  
  delete [] this->activeXAxisLabel;
  delete [] this->activeYAxisLabel;
    delete [] this->activeZAxisLabel;
  
  // Close log file.
  this->Log.close();

}

void vtkTrackingSystem::PrintSelf(ostream &os, vtkIndent indent) {

  // Connection status.
  os <<"Connection status: " << this->IsConnected << endl;

  // Number of sensors.
  os <<"Number of active sensors: " << this->NumSensors << endl;

  // Number of transmitters.
  os <<"Number of available transmitters: " << this->NumTransmitters << endl;
  os <<"\n";

  // Sensor pos/orient, prints the components in the following order: x, y, z, azi, elev,
  // roll, timestamp.
  os << "Sensor pos/orient: " << endl;
  os << "-------------------" << endl;
  for(int i=0; i < this->NumSensors; i++) {
    os << "Sensor " << i << ": " << endl;
    for(int j=0; j < this->SensorEulerAngles[i]->GetNumberOfComponents(); j++)
      os <<"  " << this->SensorEulerAngles[i]->GetComponent(i, j) << endl; 
  }
  os <<"\n";

  // Sensor time stamps.
  os << "Sensor time stamps: " << endl;
  os << "--------------------" << endl;
  for(int i=0; i < this->NumSensors; i++) 
    os << "Sensor " << i << " time stamp: " << this->SensorTimeStamps[i] << endl;
  os << "\n";

  // Sensor matrices.
  os <<"Sensor matrices: " << endl;
  os <<"-----------------" << endl;
  for(int i=0; i < this->NumSensors; i++) {
    os <<"(" << i << ")-\n";
    for(int j=0; j < 4; j++) {
      os <<this->SensorMatrices[i]->GetElement(j, 0) << " " << this->SensorMatrices[i]->GetElement(j, 1)
           << " " << this->SensorMatrices[i]->GetElement(j, 2) << " " << this->SensorMatrices[i]->GetElement(j, 3) << endl;
    }
    os <<"\n";
  }
  os <<"\n";

  // Calibration matrices.
  os <<"Calibration matrices: " << endl;
  os <<"----------------------" << endl;
  for(int i=0; i < this->NumSensors; i++) {
    os <<"(" << i << ")-\n";
    for(int j=0; j < 4; j++) {
      os <<this->CalibrationMatrices[i]->GetElement(j, 0) << " " << this->CalibrationMatrices[i]->GetElement(j, 1)
           << " " << this->CalibrationMatrices[i]->GetElement(j, 2) << " " << this->CalibrationMatrices[i]->GetElement(j, 3) << endl;
    }
    os <<"\n";
  }
  os <<"\n";

  // Registration matrix.
  os <<"Registration matrix: " << endl;
  os <<"---------------------" << endl;
  for(int j=0; j < 4; j++) {
    os <<this->RegistrationMatrix->GetElement(j, 0) << " " << this->RegistrationMatrix->GetElement(j, 1)
         << " " << this->RegistrationMatrix->GetElement(j, 2) << " " << this->RegistrationMatrix->GetElement(j, 3) << endl;
  }
  os <<"\n";

  // Locator matrices.
  os <<"Locator matrices: " << endl;
  os <<"------------------" << endl;
  for(int i=0; i < this->NumSensors; i++) {
    os <<"(" << i << ")-\n";
    for(int j=0; j < 4; j++) {
      os <<this->LocatorMatrices[i]->GetElement(j, 0) << " " << this->LocatorMatrices[i]->GetElement(j, 1)
         << " " << this->LocatorMatrices[i]->GetElement(j, 2) << " " << this->LocatorMatrices[i]->GetElement(j, 3) << endl;
    }
    os <<"\n";
  }
  os <<"\n";

}

int vtkTrackingSystem::OpenConnection() {

  /* Initialize matrices. */
  // Allocate space.
  
  this->SensorMatrices = new vtkMatrix4x4 *[this->NumSensors];
  this->SensorEulerAngles = new vtkDoubleArray *[this->NumSensors]; 
  this->CalibrationMatrices = new vtkMatrix4x4 *[this->NumSensors];
  this->LocatorMatrices = new vtkMatrix4x4 *[this->NumSensors];
  this->SensorTimeStamps = new double [this->NumSensors];

  for(int i=0; i < this->NumSensors; i++) {
    this->SensorMatrices[i] = vtkMatrix4x4::New();
    this->SensorEulerAngles[i] = vtkDoubleArray::New();
    this->SensorEulerAngles[i]->SetNumberOfComponents(6);
    this->SensorEulerAngles[i]->SetNumberOfTuples(1);
    this->CalibrationMatrices[i] = vtkMatrix4x4::New();
    this->LocatorMatrices[i] = vtkMatrix4x4::New();
  }
  // Set all to identity matrix.
  for(int i=0; i < this->NumSensors; i++) {
    this->SensorMatrices[i]->Identity();
    this->CalibrationMatrices[i]->Identity();
    this->LocatorMatrices[i]->Identity();
  }
  
  std::string tmp;
  for (int i=0; i<this->NumSensors;i++) {
    this->XAxisLabel.push_back(tmp);
    this->YAxisLabel.push_back(tmp);
        this->ZAxisLabel.push_back(tmp);
  }

  this->IsConnected = TRUE;

  return TRUE;

}

int vtkTrackingSystem::CloseConnection() {

  if (this->IsConnected == FALSE) {
    return TRUE;
  }
   //Deallocate sensor data
  for (int i=0 ; i< this->NumSensors; i++) {
    this->SensorMatrices[i]->Delete();
    this->SensorEulerAngles[i]->Delete();
    this->CalibrationMatrices[i]->Delete();
    this->LocatorMatrices[i]->Delete();
    }

  delete [] this->SensorMatrices;
  delete [] this->SensorEulerAngles;
  delete [] this->CalibrationMatrices;
  delete [] this->LocatorMatrices;
  delete [] this->SensorTimeStamps;
   
    this->SensorMatrices = NULL;
  this->CalibrationMatrices = NULL;
  this->LocatorMatrices = NULL;
  this->SensorTimeStamps = NULL;
    
    this->XAxisLabel.clear();  
    this->YAxisLabel.clear();
    this->ZAxisLabel.clear();
    
  this->IsConnected = FALSE;

  return TRUE;
  
}

void vtkTrackingSystem::Poll() {

  return;

}

void vtkTrackingSystem::UpdateLocatorMatrices() {

  // Check connection status.
  if(!this->IsConnected) return;
  
  // Refresh the sensor data.
  this->Poll();

  // Update the locator matrices.
  vtkTransform *transform = vtkTransform::New();
  transform->PostMultiply();
  for(int i=0; i < this->NumSensors; i++) {
    // Set the transform to the identity matrix.
    transform->Identity();
    // Perform Transform[identity]->Transform[point_of_interest] (point of interest (POA) in sensor reference frame)
    transform->Concatenate(this->CalibrationMatrices[i]);
    //transform->Modified();
    // Perform Transform[point_of_interest]->Transform[transmitter] (POA in transmitter frame)
    transform->Concatenate(this->SensorMatrices[i]);
    //transform->Modified();
    // Perform Transform[transmitter]->Transform[Slicer] (POA in Slicer frame)
    if (this->RegistrationTransform) {
      transform->Concatenate(this->RegistrationTransform->GetMatrix());
    }
    transform->Modified();
    // Set the locator matrix to the calculated value.
    this->LocatorMatrices[i]->DeepCopy(transform->GetMatrix());
  }
  
  // Delete the transform.
  transform->Delete();
  // Update Axis information
  this->LabelPlaneAxes();

}

void vtkTrackingSystem::LabelPlaneAxes(void) {

  double x[4] = {-1,0,0,0}; //x-axis for probes are calibrated from left to right 
  double y[4] = {0,1,0,0};
  double z[4] = {0,0,-1,0};
  double out[4];
   for(int i=0; i < this->NumSensors; i++) {

      this->LocatorMatrices[i]->MultiplyPoint(x,out);
    this->FindMaxAxis((this->XAxisLabel[i]),out);
      this->LocatorMatrices[i]->MultiplyPoint(y,out);
    this->FindMaxAxis((this->YAxisLabel[i]),out);
      this->LocatorMatrices[i]->MultiplyPoint(z,out);
          this->FindMaxAxis((this->ZAxisLabel[i]),out);
   }
}

void vtkTrackingSystem::FindMaxAxis(std::string &str,double x[4]) 
{
  //Clean current string
  str.clear();
  double xx0 = x[0]*x[0];
  double xx1 = x[1]*x[1];
  double xx2 = x[2]*x[2];
  int max =0;
  if (xx1>xx0&&xx1>xx2)
    max = 1;
  else if (xx2>xx0&&xx2>xx1)
    max = 2;
  //Cos square of 60 deg
  // If the non-principal axis angle is smaller than 
  // 60 deg, we use a two letter label.
  double cos2= 0.25;
  
  switch (max) {
  
    case 0:
      if (x[0]>=0){
         str="R";
         }
      else {
         str="L";
   }
      if ( xx1> cos2) {
          if (x[1]>=0) {
      str = str+"A";
    } else {
      str = str+"P";
    }
      } else if (xx2 > cos2) {
          if (x[2]>=0) {
      str = str+ "S";
    } else {
      str = str+ "I";
    }
      }
    break;
    case 1:
       if (x[1]>=0)
         {
         str = str +"A";
         }
       else {
          str = str + "P";
   }
       if (xx0 > cos2) {
          if (x[0]>=0) {
      str =  str + "R";
    } else {
      str = str + "L";
    }
      } else if (xx2 > cos2) {
          if (x[2]>=0) {
      str = str + "S";
    } else {
      str = str + "I";
    }
      }

       break;   
    case 2:
       if (x[2]>=0) {
          str = str + "S";
       } else {
          str = str + "I";
       }
      if (xx0>cos2) {
          if (x[0]>=0) {
      str = str + "R";
    } else {
      str = str + "L";
    }
      } else if (xx1>cos2) {
          if (x[1]>=0) {
      str = str + "A";
    } else {
      str = str + "P";
    }
      }
       break;
    }
}

char * vtkTrackingSystem::GetXAxisLabel(int sensor) {
  if (sensor>=this->NumSensors)
    return NULL;
  strcpy(activeXAxisLabel,this->XAxisLabel[sensor].c_str());
  return activeXAxisLabel;

}

char * vtkTrackingSystem::GetYAxisLabel(int sensor) {
  if (sensor>=this->NumSensors)
    return NULL;
  strcpy(activeYAxisLabel,this->YAxisLabel[sensor].c_str());
  return activeYAxisLabel;
}

char * vtkTrackingSystem::GetZAxisLabel(int sensor) {
  if (sensor>=this->NumSensors)
    return NULL;
  strcpy(activeZAxisLabel,this->ZAxisLabel[sensor].c_str());
  return activeZAxisLabel;
}

void vtkTrackingSystem::SetCalibrationMatrix(int index, vtkMatrix4x4 *matrix) {
  if (this->CalibrationMatrices && index < this->NumSensors)
     this->CalibrationMatrices[index]->DeepCopy(matrix);

}

void vtkTrackingSystem::SetCalibrationMatrixElement(int index, int i, int j, double value) {
  if (index < this->NumSensors)
    this->CalibrationMatrices[index]->SetElement(i, j, value);

}

vtkMatrix4x4 *vtkTrackingSystem::GetCalibrationMatrix(int index) {

   if (index < this->NumSensors)
       return this->CalibrationMatrices[index];
   else
       return NULL;
}

double vtkTrackingSystem::GetCalibrationMatrixElement(int index, int i, int j) {

  if (index < this->NumSensors)
    return this->CalibrationMatrices[index]->GetElement(i, j);
  else
    return 0.0;
}

void vtkTrackingSystem::SetRegistrationTransform(vtkLinearTransform *t) {
   if (this->RegistrationTransform == t)
     {
     return;
     }
   if (this->RegistrationTransform)
    {
    this->RegistrationTransform->Delete();
    }

    if (t)
      t->Register(this);
    this->RegistrationTransform = t;
   
    this->SetRegistrationMatrix(t->GetMatrix());
  
    this->Modified();
}     
void vtkTrackingSystem::SetRegistrationMatrix(vtkMatrix4x4 *matrix) {
  
    if (this->RegistrationMatrix == matrix)
    {
    return;
    }

    if (this->RegistrationMatrix)
    {
    this->RegistrationMatrix->Delete();
    }

    if (matrix)
      matrix->Register(this);
    this->RegistrationMatrix = matrix;

    this->Modified();
}

void vtkTrackingSystem::SetRegistrationMatrixElement(int i, int j, double value) {

  if (this->RegistrationMatrix)
     this->RegistrationMatrix->SetElement(i, j, value);

}

vtkMatrix4x4 *vtkTrackingSystem::GetRegistrationMatrix(){

  return this->RegistrationMatrix;

}

double vtkTrackingSystem::GetRegistrationMatrixElement(int i, int j) {

  if (this->RegistrationMatrix)
    return this->RegistrationMatrix->GetElement(i, j);
  else
    return 0;
}

void vtkTrackingSystem::SetSensorMatrix(int index, vtkMatrix4x4 *matrix) {
  if (this->SensorMatrices && index < this->NumSensors)
     this->SensorMatrices[index]->DeepCopy(matrix);

}

vtkMatrix4x4 *vtkTrackingSystem::GetSensorMatrix(int index) {

  if (index < this->NumSensors)
    return this->SensorMatrices[index];
  else 
    return NULL;
}

vtkMatrix4x4 *vtkTrackingSystem::GetLocatorMatrix(int index) {
  
  if (index < this->NumSensors)
    return this->LocatorMatrices[index];
  else 
    return NULL;
}

vtkDoubleArray *vtkTrackingSystem::GetSensorEulerAngles(int index) {

  if (index < this->NumSensors)
    return this->SensorEulerAngles[index];
  else 
    return NULL;
}

double vtkTrackingSystem::GetSensorTimeStamp(int index) {
  
  if (index < this->NumSensors)        
    return this->SensorTimeStamps[index];
  else
    return 0.0;
}  

void vtkTrackingSystem::BuildMatrixFromEulerAngles(vtkDoubleArray **all_data, int index, vtkMatrix4x4 *matrix) {

  vtkDoubleArray *row_data = vtkDoubleArray::New();
  row_data->SetNumberOfComponents(6);
  row_data->SetNumberOfTuples(1);

  // Assign components.
  row_data->SetComponent(0, 0, all_data[index]->GetComponent(0, 0)); // x
  row_data->SetComponent(0, 1, all_data[index]->GetComponent(0, 1)); // y
  row_data->SetComponent(0, 2, all_data[index]->GetComponent(0, 2)); // z
  row_data->SetComponent(0, 3, (all_data[index]->GetComponent(0, 3) )); // alpha:rad or degrees
  row_data->SetComponent(0, 4, (all_data[index]->GetComponent(0, 4) )); // beta: rad or degrees
  row_data->SetComponent(0, 5, (all_data[index]->GetComponent(0, 5) )); // gamma: rad or degrees
  // Calculate matrix.
  BuildMatrixFromEulerAngles(row_data, matrix);

  row_data->Delete();

}

void vtkTrackingSystem::BuildMatrixFromEulerAngles(vtkDoubleArray *angles, vtkMatrix4x4 *matrix) {

  double mat[4][4];
  double x,y,z,alpha,beta,gamma;
  
  if (angles->GetNumberOfComponents()<6)
    vtkErrorMacro("6 components are neccesary: x y z alpha beta gamma");

  x = angles->GetComponent(0,0);
  y = angles->GetComponent(0,1);
  z = angles->GetComponent(0,2);

  alpha = angles->GetComponent(0,3);
  beta = angles->GetComponent(0,4);
  gamma = angles->GetComponent(0,5);

  this->BuildMatrixFromEulerAngles(x,y,z,alpha,beta,gamma,mat);
    
  for(int i=0;i<4;i++) {
    for(int j=0;j<4;j++) {
      matrix->SetElement(i,j,mat[i][j]);
    }
  }

}

void vtkTrackingSystem::BuildMatrixFromEulerAngles(double x, double y, double z, double alpha, double beta, double gamma, double mat[4][4]) {

  if (this->EulerAnglesInDegrees) {
     alpha = alpha * PI/180; // alpha: deg->rad 
     beta = beta * PI/180;  // beta: deg->rad
     gamma = gamma *PI/180; // gamma: deg->rad
  }   

  // Rotation.
  mat[0][0] = cos(alpha) * cos(beta);
  mat[0][1] = cos(alpha) * sin(beta) * sin(gamma) - sin(alpha) * cos(gamma);
  mat[0][2] = cos(alpha) * sin(beta) * cos(gamma) + sin(alpha) * sin(gamma);
  mat[1][0] = sin(alpha) * cos(beta);
  mat[1][1] = sin(alpha) * sin(beta) * sin(gamma) + cos(alpha) * cos(gamma);
  mat[1][2] = sin(alpha) * sin(beta) *cos(gamma) - cos(alpha) * sin(gamma);
  mat[2][0] = -sin(beta);
  mat[2][1] = cos(beta) * sin(gamma);
  mat[2][2] = cos(beta) * cos(gamma);

  // Translation.
  mat[0][3] = x;
  mat[1][3] = y;
  mat[2][3] = z;

  mat[3][0] = 0;
  mat[3][1] = 0;
  mat[3][2] = 0;
  mat[3][3] = 1.0;

}

void vtkTrackingSystem::BuildEulerAnglesFromMatrix(vtkMatrix4x4 *matrix, vtkDoubleArray *angles) {

  double mat[4][4];
  double x,y,z,alpha,beta,gamma;

    for(int i=0;i<4;i++) {
    for(int j=0;j<4;j++) {
      mat[i][j]=matrix->GetElement(i,j);
    }
  }

   this->BuildEulerAnglesFromMatrix(mat,x,y,z,alpha,beta,gamma);
   angles->SetNumberOfComponents(6);
   angles->SetNumberOfTuples(1);
   angles->SetComponent(0,0,x);
   angles->SetComponent(0,1,y);
   angles->SetComponent(0,2,z);
   angles->SetComponent(0,3,alpha);
   angles->SetComponent(0,4,beta);
   angles->SetComponent(0,5,gamma);

}

void vtkTrackingSystem::BuildEulerAnglesFromMatrix(double mat[4][4], double& x, double& y, double&z, double &alpha, double &beta, double &gamma) {

  x = mat[0][3];
  y = mat[1][3];
  z = mat[2][3];

  beta = asin(-mat[2][0]);
  alpha = atan2(mat[1][0],mat[0][0]);
  gamma = atan2(mat[2][1],mat[2][2]);

  // NORTH POLE
  if (mat[2][0]==-1) {
   beta = PI/2;
   alpha = 0;
   gamma = atan2(mat[0][1],mat[0][2]);
  }

  // SOUTH POLE
  if (mat[2][0]==1) {
   beta = -PI/2;
   alpha = 0;
   gamma =  atan2(-mat[1][2],mat[1][1]);
  }

}

