/*=========================================================================

Program:   Visualization Toolkit
Module:    $RCSfile: vtkMiniBirdTrackingSystem.cxx,v $
Language:  C++
Date:      $Date: 2007/07/12 16:24:29 $
Version:   $Revision: 1.11 $

=========================================================================*/

#include "vtkMiniBirdTrackingSystem.h"
#include <time.h>

vtkCxxRevisionMacro(vtkMiniBirdTrackingSystem, "$Revision: 1.11 $");
vtkStandardNewMacro(vtkMiniBirdTrackingSystem);

vtkMiniBirdTrackingSystem::vtkMiniBirdTrackingSystem() {

  // Set mode of operation.
  this->ModeOfOperation = NOT_SET;
  
  // Set default baud rate.
  this->BaudRate = DEFAULT_BAUD_RATE;
  this->COMPort = 1;
  
}

vtkMiniBirdTrackingSystem::~vtkMiniBirdTrackingSystem() {
  
}

void vtkMiniBirdTrackingSystem::PrintSelf(ostream &os, vtkIndent indent) {
  
  // Mode of operation.
  os << "Mode of operation: " << this->ModeOfOperation << endl;
  
  // Current baud rate.
  os << "Baud rate: " << this->BaudRate << endl;
  
  // Print parent information.
  this->Superclass::PrintSelf(os, indent);
  
}

int vtkMiniBirdTrackingSystem::OpenConnection() {
  
  int error; // holds returned error value
  WORD *com_ports; // array of COM port numbers for the sensors
  
  if (this->GetIsConnected()) {
  return TRUE;
  }
  
  /* Make sure settings are correct. */
  // Check that mode of operation is set.
  if(this->ModeOfOperation == NOT_SET) {
  this->Log << "MiniBird: Connection request attempted without setting mode of operation.\n";
  return FALSE;
  }
  // Check that number of sensors is set.
  if(this->NumSensors == 0) {
  this->Log << "MiniBird: Connection request attempted without setting the number of sensors.\n";
  return FALSE;
  }
  // Check that mode of operation is compatible with number of birds.
  //if(this->ModeOfOperation == STANDALONE && this->NumSensors > 1) {
  //  this->Log << "MiniBird: Mode requested is not compatible with number of sensors specified.\n";
  //  return FALSE;
  //}
  
  
  /* Initialize the system. */
  com_ports = new WORD [this->NumSensors];
  for (int i=0; i< this->NumSensors; i++) {
  com_ports[i] = this->COMPort;
  }
  error = birdRS232WakeUp((int) GROUP_ID, (bool) this->ModeOfOperation, this->NumSensors, com_ports, (DWORD) this->BaudRate, (DWORD) 1000, (DWORD) 1000); 
  delete [] com_ports;
  if(error == FAILED) {
  this->Log << "MiniBird: Initialization error.\n";
  return FALSE;
  } else {
  this->Log << "MiniBird: Initialization successful.\n";
  }
  
  /* Get the system configuration. */
  error = birdGetSystemConfig((int) GROUP_ID, &(this->SystemConfig));
  if(error == FAILED) {
  this->Log << "MiniBird: Could not get system configuration.\n";
  return FALSE;
  } 
  
  /* Get the sensor configurations. */
  // Allocate space for configurations and future sensor data.
  this->SensorConfigs = new BIRDDEVICECONFIG[this->NumSensors];
  // Set sensors' configurations.
  for(int i=0; i < this->NumSensors; i++) {   
  // Get configuration: If in STANDALONE mode, the index requested is 
  // ignored, although it is 0; however; if in FOB mode, the index begins 
  // at 1, hence the (i+1).
  error = birdGetDeviceConfig((int) GROUP_ID, i+1, &this->SensorConfigs[i]);
  // Check for failure.
  if(error ==  FAILED) {
  this->Log << "MiniBird: Could not get sensor " << i+1 << "'s configuration.\n";
  return FALSE;
  } else {
  // Set data format type for sensor data.
  this->SensorConfigs[i].byDataFormat = BDF_POSITIONANGLES;
  //this->SensorConfigs[i].byDataFormat = BDF_POSITIONMATRIX;
  birdSetDeviceConfig((int) GROUP_ID, i+1, &(this->SensorConfigs[i]));
  }
  }
  
  // Set connection status.
  this->IsConnected = TRUE;
  
  // Allocate matrices. 
  this->Superclass::OpenConnection();
  
  return TRUE;
  
}

int vtkMiniBirdTrackingSystem::CloseConnection() {
  
  if (!this->GetIsConnected()) {
  return TRUE;
  }
  
  // Shutdown the system.
  birdShutDown((int) GROUP_ID);
  
  // Set connection status.
  this->IsConnected = FALSE;
  
  // Reset number of sensors, transmitters.
  this->NumSensors = 0;
  this->NumTransmitters = 0;
  
  this->Superclass::CloseConnection();
  
  return TRUE;
  
}

void vtkMiniBirdTrackingSystem::Poll() {

  BIRDFRAME sensors_data; // holds data for all sensors and a timestamp
  BIRDREADING reading; // individual sensor's data
  double pos_scale; 

  // Check connection status.
  if(this->IsConnected == FALSE) {
  this->Log << "MiniBird: Attempt made to refresh data without being connected.\n";
  return;
  }

  // Initiate the frame capture.
  birdStartSingleFrame((int) GROUP_ID);

  // Wait until the frame is ready.
  while(!birdFrameReady((int) GROUP_ID)) {};

  // Obtain the frame.
  birdGetFrame((int) GROUP_ID, &sensors_data);

  /* Update sensor information for all active sensors. */  
  for(int i=0; i < this->NumSensors; i++) {
  // Get sensor's data.
  if(this->ModeOfOperation == STANDALONE)
    reading = sensors_data.reading[i];
  else if(this->ModeOfOperation == FOB)
    reading = sensors_data.reading[i+1];

  // Set scaling.
  pos_scale = this->SensorConfigs[i].wScaling;

  /* Set time stamp for acquisition. */
  this->SensorTimeStamps[i] = sensors_data.dwTime;
  this->SensorTimeStamps[i] = (int) clock();
    
  if (this->SensorConfigs[i].byDataFormat == BDF_POSITIONANGLES) {
  //* Set angles */
  // Set position information.
  int pos_scale = this->SensorConfigs[i].wScaling;
  this->SensorEulerAngles[i]->SetComponent(0, 0, (double) (reading.position.nX * pos_scale / 32767.0) * (INCH_TO_CM*10) );
  this->SensorEulerAngles[i]->SetComponent(0, 1, (double) (reading.position.nY * pos_scale / 32767.0) * (INCH_TO_CM*10) );
  this->SensorEulerAngles[i]->SetComponent(0, 2, (double) (reading.position.nZ * pos_scale / 32767.0) * (INCH_TO_CM*10) );

  // Set orientation information.
  this->SensorEulerAngles[i]->SetComponent(0, 3, (double) reading.angles.nAzimuth * 180./ 32767.0);
  this->SensorEulerAngles[i]->SetComponent(0, 4, (double) reading.angles.nElevation * 180./ 32767.0);
  this->SensorEulerAngles[i]->SetComponent(0, 5, (double) reading.angles.nRoll * 180./ 32767.0);
    
  this->SetEulerAnglesInDegrees();
  this->BuildMatrixFromEulerAngles(this->SensorEulerAngles[i],this->SensorMatrices[i]);
    
  } else if(this->SensorConfigs[i].byDataFormat == BDF_POSITIONMATRIX) {
  /* Set matrix. */
  // Rotation.
  for(int j=0; j < 3; j++) {
  for(int k=0; k < 3; k++) {
  this->SensorMatrices[i]->SetElement(j, k, (double) reading.matrix.n[k][j] / 32767.0);
  }
  }
  // Position.
  this->SensorMatrices[i]->SetElement(0, 3, (reading.position.nX * pos_scale / 32767.0) * (INCH_TO_CM*10));
  this->SensorMatrices[i]->SetElement(1, 3, (reading.position.nY * pos_scale / 32767.0) * (INCH_TO_CM*10));
  this->SensorMatrices[i]->SetElement(2, 3, (reading.position.nZ * pos_scale / 32767.0) * (INCH_TO_CM*10));

  this->BuildEulerAnglesFromMatrix(this->SensorMatrices[i],this->SensorEulerAngles[i]);
  }  


  } 
}
