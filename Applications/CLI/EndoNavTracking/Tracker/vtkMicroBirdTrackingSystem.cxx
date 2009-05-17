/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkMicroBirdTrackingSystem.cxx,v $
  Language:  C++
  Date:      $Date: 2008/05/27 16:15:33 $
  Version:   $Revision: 1.12 $

=========================================================================*/

#include "vtkMicroBirdTrackingSystem.h"
#include <time.h>

vtkCxxRevisionMacro(vtkMicroBirdTrackingSystem, "$Revision: 1.12 $");
vtkStandardNewMacro(vtkMicroBirdTrackingSystem);

vtkMicroBirdTrackingSystem::vtkMicroBirdTrackingSystem() {
  
  // Set number of attached sensors.
  this->NumAttachedSensors = 0;
  
  // Set number of attached transmitters.
  this->NumAttachedTransmitters = 0;
  
  //Tracking system set up variables
  this->MeasurementRate = 68.3;
}

vtkMicroBirdTrackingSystem::~vtkMicroBirdTrackingSystem() {

}

void vtkMicroBirdTrackingSystem::PrintSelf(ostream& os, vtkIndent indent) {
  
  // Number of attached sensors.
  os << "Number of attached sensors: " << this->NumAttachedSensors << endl;

  // Number of attached transmitters.
  os << "Number of attached transmitters: " << this->NumAttachedTransmitters << endl;

  // Print parent information.
  this->Superclass::PrintSelf(os, indent);

}

int vtkMicroBirdTrackingSystem::OpenConnection() {

  int error; // holds returned error codes
  
  if (this->GetIsConnected()) {
    return TRUE;
  }

  /* Initialize the system. */
  error = InitializeBIRDSystem();
  if(error != BIRD_ERROR_SUCCESS) {
    this->Log << "MicroBird: Initialization error.\n";
    this->Log << "MicroBird: Error code ==> " << error << "\n";
    return FALSE;
  } else {
    this->Log << "MicroBird: Initialization successful.\n";
  }

  /* Get the system configuration. */
  error = GetBIRDSystemConfiguration(&this->SystemConfig);
  if(error != BIRD_ERROR_SUCCESS) {
    this->Log << "MicroBird: Could not get system configuration.\n";
    this->Log << "MicroBird: Error code => " << error << "\n";
    return FALSE;
  } else {
    // Set the number of sensor ports.
    this->NumSensors = this->SystemConfig.numberSensors;
    // Set the number of transmitter ports.
    this->NumTransmitters = this->SystemConfig.numberTransmitters;
    // Set the measurement rate.
    double measurement_rate = this->MeasurementRate;
    error = SetSystemParameter(MEASUREMENT_RATE, &(measurement_rate), sizeof(double));
    if(error != BIRD_ERROR_SUCCESS) { 
      this->Log << "MicroBird: Not able to set MEASUREMENT_RATE.\n";
      this->Log << "MicroBird: Error code => " << error << "\n";
      return FALSE;
    }
    double maximum_range = 36;
    error = SetSystemParameter(MAXIMUM_RANGE, &(maximum_range), sizeof(double));
        if(error != BIRD_ERROR_SUCCESS) { 
      this->Log << "MicroBird: Not able to set MEASUREMENT_RATE.\n";
      this->Log << "MicroBird: Error code => " << error << "\n";
      return FALSE;
    }
    double powerline =60.0;
        error = SetSystemParameter(POWER_LINE_FREQUENCY, &(powerline), sizeof(double));
        if(error != BIRD_ERROR_SUCCESS) { 
      this->Log << "MicroBird: Not able to set POWER_LINE_FREQUENCY.\n";
      this->Log << "MicroBird: Error code => " << error << "\n";
      return FALSE;
    }
    // Set measurement units to metric.    
    BOOL metric = 1;
    error = SetSystemParameter(METRIC, &(metric), sizeof(BOOL)); 
    if(error != BIRD_ERROR_SUCCESS) {
      this->Log << "MicroBird: Not able to set measurement units to METRIC.\n";
      this->Log << "MicroBird: Error code => " << error << "\n";
      return FALSE;
    }

  }

  /* Get transmitter port configurations. */
  // Allocate space for configurations.
  this->TransmitterConfigs = new TRANSMITTER_CONFIGURATION[this->NumTransmitters];
  // Set transmitters' configurations.
  for(int i=0; i < this->NumTransmitters; i++) {
    error = GetTransmitterConfiguration(i, &this->TransmitterConfigs[i]);
    if(error != BIRD_ERROR_SUCCESS) {
      this->Log << "MicroBird: Could not get transmitter port " << (i+1) << "'s configuration.\n";
      this->Log << "MicroBird: Error code => " << error << "\n";
      return FALSE;
    } else {
      // Check if a transmitter is attached to the port and adjust number
      // of active transmitters accordingly.
      if(this->TransmitterConfigs[i].attached)
        this->NumAttachedTransmitters++;
    }
  }

  /* Get sensor port configurations. */
  // Allocate space for configurations.
  this->SensorConfigs = new SENSOR_CONFIGURATION[this->NumSensors];
  // Set sensors' configurations.
  for(int i=0; i < this->NumSensors; i++) {
    error = GetSensorConfiguration(i, &this->SensorConfigs[i]);
    if(error != BIRD_ERROR_SUCCESS) {
      this->Log << "MicroBird: Could not get sensor port " << (i+1) << "'s configuration.\n";
      this->Log << "MicroBird: Error code => " << error << "\n";
      return FALSE;
    } else {
      // Set data format type for all sensor data.
      DATA_FORMAT_TYPE format = DOUBLE_ALL_TIME_STAMP;
      error = SetSensorParameter(i, DATA_FORMAT, &format, sizeof(DATA_FORMAT_TYPE));
      if(error != BIRD_ERROR_SUCCESS) {
        this->Log << "MicroBird: Could not set sensor data format type.\n";
        this->Log << "MicroBird: Error code => " << error << "\n";
        return FALSE;
      } 
      // Check if a sensor is attached to the port and adjust number
      // of active sensors accordingly.
      if(this->SensorConfigs[i].attached)
        this->NumAttachedSensors++;
    }
  }

  /* Activate first attached transmitter found. */
  for(int i=0; i < this->NumTransmitters; i++) {
    if(this->TransmitterConfigs[i].attached) {
      // Attempt to set to active transmitter.
      error = SetSystemParameter(SELECT_TRANSMITTER, &i, sizeof(short));
      if(error != BIRD_ERROR_SUCCESS) { 
        this->Log << "MicroBird: Not able to set transmitter port " << (i+1) << " to active transmitter.\n";
        this->Log << "MicroBird: Error code => " << error << "\n";
        // Check if all transmitters have been exhausted.
        if(i == this->SystemConfig.numberTransmitters-1) { 
          this->Log << "MicroBird: All transmitters exhausted.\n";
          return FALSE;
        }
      }
    }
  }

  // Set connection status.
  this->IsConnected = TRUE;
  
  // Allocate matrices.
  this->Superclass::OpenConnection();

  return TRUE;

}

int vtkMicroBirdTrackingSystem::CloseConnection() {

  int error; // holds returned error codes

  if (!this->GetIsConnected()) {
          return TRUE;
  }

  /* Shut down the system. */
  error = CloseBIRDSystem();

  if(error != BIRD_ERROR_SUCCESS) {
    this->Log << "MicroBird: System unable to shutdown properly.\n";
    this->Log << "MicroBird: Error code => " << error << "\n";
    return FALSE;
  }

  // Set connection status.
  this->IsConnected = FALSE;

  // Reset number of sensors, transmitters.
  this->NumSensors = 0;
  this->NumTransmitters = 0;

  // Reset number of active sensors, 
  // transmitters.
  this->NumAttachedSensors = 0;
  this->NumAttachedTransmitters = 0;
    
  // Deallocate data.
  this->Superclass::CloseConnection();

  return TRUE;
    
}

void vtkMicroBirdTrackingSystem::Poll() {

  int error; // holds returned error codes
  DOUBLE_ALL_TIME_STAMP_RECORD sensor_data; // hold sensor's data and a timestamp

  // Check connection status.
  if(this->IsConnected == FALSE) {
    this->Log << "MicroBird: Attempt made to refresh data without being connected.\n"; 
    return;
  }

  // Update sensor information for all active sensors.
  for(int i=0; i < this->NumSensors; i++) {  
    if(this->SensorConfigs[i].attached) {
      error = GetAsynchronousRecord(i, &sensor_data, sizeof(DOUBLE_ALL_TIME_STAMP_RECORD));
      if(error != BIRD_ERROR_SUCCESS) {
        this->Log << "MicroBird: Unable to acquire sensor port" << (i+1) << "'s data.\n";
        this->Log << "MicroBird: Error code => " << error << "\n";
      } else {
        
        /* Set time stamp for acquisition. */
        //this->SensorTimeStamps[i] = sensor_data.time;
              this->SensorTimeStamps[i] = (int) clock();
        
        /* Set matrix. */
        // Rotation.
        for(int j=0; j < 3; j++) {
          for(int k=0; k < 3; k++) {
            this->SensorMatrices[i]->SetElement(j, k, sensor_data.s[k][j]);
          }
        }
        // Position.
        this->SensorMatrices[i]->SetElement(0, 3, sensor_data.x);
        this->SensorMatrices[i]->SetElement(1, 3, sensor_data.y);
        this->SensorMatrices[i]->SetElement(2, 3, sensor_data.z);

        //* Set angles */
        // Set position information.
        this->SensorEulerAngles[i]->SetComponent(0, 0, sensor_data.x); // x-axis
        this->SensorEulerAngles[i]->SetComponent(0, 1, sensor_data.y); // y-axis
        this->SensorEulerAngles[i]->SetComponent(0, 2, sensor_data.z); // z-axis

        // Set orientation information.
        this->SensorEulerAngles[i]->SetComponent(0, 3, sensor_data.a); // azimuth = alpha
        this->SensorEulerAngles[i]->SetComponent(0, 4, sensor_data.e); // elevation = beta
        this->SensorEulerAngles[i]->SetComponent(0, 5, sensor_data.r); // roll = gamma 

      }
       
    } 
  }

}


  

      



  


