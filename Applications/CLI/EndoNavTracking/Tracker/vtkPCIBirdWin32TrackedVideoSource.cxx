/*=========================================================================

Program:   Visualization Toolkit
Module:    $RCSfile: vtkPCIBirdWin32TrackedVideoSource.cxx,v $
Language:  C++
Date:      $Date: 2007/07/12 16:24:29 $
Version:   $Revision: 1.13 $

Copyright (c) 1993-2001 Ken Martin, Will Schroeder, Bill Lorensen 
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

* Neither name of Ken Martin, Will Schroeder, or Bill Lorensen nor the names
of any contributors may be used to endorse or promote products derived
from this software without specific prior written permission.

* Modified source versions must be plainly marked as such, and must not be
misrepresented as being the original software.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=========================================================================*/

#include "vtkPCIBirdWin32TrackedVideoSource.h"
#include "vtkMultiThreader.h"
#include "vtkMutexLock.h"
#include "vtkTimerLog.h"

//----------------------------------------------------------------------------
vtkPCIBirdWin32TrackedVideoSource* vtkPCIBirdWin32TrackedVideoSource::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkPCIBirdWin32TrackedVideoSource");
  if(ret)
    {
    return (vtkPCIBirdWin32TrackedVideoSource *)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkPCIBirdWin32TrackedVideoSource;
}

//----------------------------------------------------------------------------
vtkPCIBirdWin32TrackedVideoSource::vtkPCIBirdWin32TrackedVideoSource()
{
  //! Initiliaze the calibration matrices of each tool
  for(int i=0;i < NUM_TOOLS;i++) {
  this->m_CalibrationMatrix[i]= vtkMatrix4x4::New();
  this->m_CalibrationMatrix[i]->Identity();
    
  this->m_LocatorMatrix[i] = vtkMatrix4x4::New();
  this->m_LocatorMatrix[i]->Identity();
  }
  
  //! Initialize the registration matrix
  this->m_RegistrationMatrix = vtkMatrix4x4::New();
  this->m_RegistrationMatrix->Identity();

  //! Initilialize the physical sensors matrices ( Tsensor->loc ) and the virtual sensors matrices ( Ttip->ct) 
  for(int i=0;i < MAX_SENSORS;i++) {
  this->m_SensorMatrixInLoc[i]= vtkMatrix4x4::New();
  this->m_SensorMatrixInLoc[i]->Identity();
  this->m_ProbeBirdAngles[i]=vtkDoubleArray::New();
  this->m_ProbeBirdAngles[i]->SetNumberOfTuples(3);
  this->m_SensorTime[i] = 0.0 ;
  }
    
  // Initialize recording variables
  this->MaxNumberOfFrames = 1;
  this->RecordingPort = 0;
  this->SensorBuffer = vtkDoubleArray::New();
  this->VideoBuffer = vtkImageData::New();
  this->VideoBuffer->SetScalarTypeToUnsignedChar();

  // Set two thread: one for video and other for tracking.
  //this->PlayerThreader->SetNumberOfThreads(2);

  this->RecordingUS = 0;
  this->StopFlag = 0;
  this->Doppler = 0;

  this->m_LogFile    = fopen("C:/LapUSdata/PCIBirdConnection.txt","w");
  this->m_DataFile  = fopen("C:/LapUSdata/LocalizationData.txt","w");
}

//----------------------------------------------------------------------------
vtkPCIBirdWin32TrackedVideoSource::~vtkPCIBirdWin32TrackedVideoSource()
{
  //Release Bird
  this->CloseConnection();
  
  //! Delete all the calibration matrices
  for(int i=0;i < NUM_TOOLS;i++) {
  this->m_CalibrationMatrix[i]->Delete();
  this->m_LocatorMatrix[i]->Delete();
  }
  //! Delete the registration matrix
  this->m_RegistrationMatrix->Delete();

  for(int i=0;i < MAX_SENSORS;i++) {
  this->m_SensorMatrixInLoc[i]->Delete();
  this->m_ProbeBirdAngles[i]->Delete();
  }

  // Delete buffers
  this->SensorBuffer->Delete();
  this->VideoBuffer->Delete();
  
  fclose(m_LogFile) ;
  fclose(m_DataFile);
    
  //this->ReleaseSystemResources();
  //We should not call this, then is call twice--> BUG
  //this->vtkWin32VideoSource::~vtkWin32VideoSource();
}

//----------------------------------------------------------------------------
void vtkPCIBirdWin32TrackedVideoSource::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}

//----------------------------------------------------------------------------
int vtkPCIBirdWin32TrackedVideoSource::OpenConnection(int port, int baud, int numbirds, int video)
{  

  int l_IsConnected = 1;
  char* l_message ;

  // --- INITIALIZE SYSTEM
  int l_ErrorCode = InitializeBIRDSystem();  
  if (l_ErrorCode == BIRD_ERROR_SUCCESS) 
    {
    l_IsConnected = 1 ;
    l_message = "InitializeBirdSystem():OK \n" ;
    fwrite(l_message,1,50, m_LogFile);
    }  
  else
    {
    l_message = "InitializeBirdSystem():ERROR\n" ;
    fwrite(l_message,1,50, m_LogFile);
    l_IsConnected = 0 ;
    if(l_ErrorCode == BIRD_ERROR_NO_DEVICES_FOUND)
      {
      l_message = "BIRD_ERROR_NO_DEVICES_FOUND" ;
      fwrite(l_message,1,50, m_LogFile);
      }
    }
  
  // --- GET SYSTEM CONFIGURATION
  l_ErrorCode = GetBIRDSystemConfiguration(&m_SystemConfig);  
  
  if (l_ErrorCode == BIRD_ERROR_SUCCESS)
    {  
    l_message = "GetBirdSystemConfiguration() --> OK" ;
    fwrite(l_message,1,50, m_LogFile);
    
    //! SET THE MEASUREMENT RATE

    double l_rate = 78.0 ;
    SetSystemParameter(MEASUREMENT_RATE, &l_rate, sizeof(l_rate));

    // Set the Metric
    BOOL l_Buffer = 1 ;
    BOOL *l_pBuffer = &l_Buffer;
    //l_ErrorCode = GetSystemParameter(METRIC, l_pBuffer, sizeof(l_Buffer));
    if(l_ErrorCode == BIRD_ERROR_SUCCESS )
      {  
      l_IsConnected = 1 ;        
      SetSystemParameter(METRIC, l_pBuffer, sizeof(l_Buffer));
        
      }
    else{        
    l_IsConnected = 0 ;
    l_message = "ERROR ------GetSystemParameter()" ;
    fwrite(l_message,1,50, m_LogFile);
    }
    }
  else
    {    
    l_IsConnected = 0 ;  
    l_message = "ERROR -----GetBirdSystemConfiguration()" ;
    fwrite(l_message,1,50, m_LogFile);
    }
  
  // --- GET SENSOR CONFIGURATIONS
  for(int i=0;i<m_SystemConfig.numberSensors;i++) 
    {
    l_ErrorCode = GetSensorConfiguration(i, &m_SensorConfig[i]);
    l_message = "GetSensorConfiguration()----> OK" ;
    fwrite(l_message,1,50, m_LogFile);
    if(l_ErrorCode==BIRD_ERROR_SUCCESS){      
    // old 
    //DATA_FORMAT_TYPE l_format    = DOUBLE_POSITION_MATRIX_TIME_STAMP ;
      
    // new
    DATA_FORMAT_TYPE l_format    = DOUBLE_ALL ;
    HEMISPHERE_TYPE     l_Hemisphere  =  FRONT ;
    l_ErrorCode= SetSensorParameter(i,DATA_FORMAT,&l_format,sizeof(l_format));
    if(l_ErrorCode==BIRD_ERROR_SUCCESS)
      {
      l_IsConnected = 1 ;
      }
    else
      {
      l_IsConnected = 0 ;
      }
    }
    else
      {
      l_IsConnected = 0 ;  
      l_message = "ERROR ----> GetSensorConfiguration()" ;
      fwrite(l_message,1,50, m_LogFile);
      }
    }
  
  // --- GET TRANSMITTER CONFIGURATIONS
  for(int i=0;i<m_SystemConfig.numberTransmitters;i++) {
  l_ErrorCode = GetTransmitterConfiguration(i, &m_TransmitterConfig[i]);
  if(l_ErrorCode==BIRD_ERROR_SUCCESS) {
  l_IsConnected = 1 ;  
  l_message = "GetTransmitterConfiguration()---> OK" ;
  fwrite(l_message,1,50, m_LogFile);
  }
  else
    {
    l_IsConnected = 0 ;  
    l_message = "ERROR ----> GetTransmitterConfiguration()" ;
    fwrite(l_message,1,50, m_LogFile);
    }
  } 
  
  //! SEARCH FOR THE FIRST TRANSMITTER ATTACHED AND TURN IT ON
  for(short i=0;i<m_SystemConfig.numberTransmitters;i++) {

  if(m_TransmitterConfig[i].attached){
  l_ErrorCode = SetSystemParameter(SELECT_TRANSMITTER, &i, sizeof(i));
      
  if(l_ErrorCode==BIRD_ERROR_SUCCESS) {
  l_IsConnected = 1 ;  
  l_message = "SetSystemParameter()---> OK" ;
  fwrite(l_message,1,50, m_LogFile);
  }
  else
    {
    l_IsConnected = 0 ;  
    l_message = "ERRROR SetSystemParameter()---> OK" ;
    fwrite(l_message,1,50, m_LogFile);
    }
  break ;
  }
  } 
  Sleep(1000);

  return l_IsConnected ;
}

//----------------------------------------------------------------------------
void vtkPCIBirdWin32TrackedVideoSource::Poll() {

  int l_ErrorCode = 0;

  
  //! Set one acquisition per measurement cycle
  clock_t l_goal      ;
  clock_t l_wait = 0   ;  // 10 ms delay
  l_goal = l_wait + clock();

  char * l_message ;
  //! DATA FORMAT
  
  // old
  //DOUBLE_POSITION_MATRIX_TIME_STAMP_RECORD l_Record ;
  //DOUBLE_POSITION_MATRIX_TIME_STAMP_RECORD* l_pRecord =&l_Record;

  // video acquisition 
  this->vtkWin32VideoSource::Grab();
  this->vtkImageSource::UpdateInformation();

  //new 
  DOUBLE_ALL_RECORD l_Record ;
  DOUBLE_ALL_RECORD * l_pRecord =&l_Record;
 
  while(l_goal>clock());

  // set up delay for next loop
  //l_goal = l_wait + clock();

  // Reading on each port
  
  for(int p = 0 ;p < m_SystemConfig.numberSensors ; p++)
    {
    if(m_SensorConfig[p].attached)
      {    
      l_ErrorCode = GetAsynchronousRecord(p,l_pRecord,sizeof(l_Record));
      if(l_ErrorCode == BIRD_ERROR_SUCCESS)
        {  
        this->m_SensorTime[p] = (int) clock();
        // Save the position and orientation of the sensor in Rloc
        for(int i=0;i<3;i++) {
        for(int j=0;j<3;j++) {
        this->m_SensorMatrixInLoc[p]->SetElement(i,j,l_Record.s[j][i]);
        //double l_value = this->m_SensorMatrixInLoc[p]->GetElement(j,i);
        //fprintf(m_DataFile,"%f\n",l_value);
        }
        }

        this->m_SensorMatrixInLoc[p]->SetElement(0,3,l_Record.x);
        this->m_SensorMatrixInLoc[p]->SetElement(1,3,l_Record.y);
        this->m_SensorMatrixInLoc[p]->SetElement(2,3,l_Record.z);
        this->m_SensorMatrixInLoc[p]->SetElement(3,0,(double)0.0);
        this->m_SensorMatrixInLoc[p]->SetElement(3,1,(double)0.0);
        this->m_SensorMatrixInLoc[p]->SetElement(3,2,(double)0.0);
        this->m_SensorMatrixInLoc[p]->SetElement(3,3,(double)1.0);  
                
        // get the orientation angles of the sensor
        this->m_ProbeBirdAngles[p]->SetValue(0,l_Record.a) ; // azimuth
        this->m_ProbeBirdAngles[p]->SetValue(1,l_Record.e) ; // elevation
        this->m_ProbeBirdAngles[p]->SetValue(2,l_Record.r); // roll  
        
        //fprintf(m_DataFile,"%f\n",l_Record.x);
        //fprintf(m_DataFile,"%f\n",l_Record.y);
        //fprintf(m_DataFile,"%f\n",l_Record.z);
          
        //! Compute the transformation from transmitter to CT frame
        this->UpdateLocatorMatrix(p);
        }

      else
        {
        l_message = "NO DATA COLLECTED" ;
        fwrite(l_message,1,30, m_DataFile);
        }
       
      }
    }

}

//----------------------------------------------------------------------------
// platform-independent sleep function
static inline void vtkSleep2(double duration)
{
  duration = duration; // avoid warnings
  // sleep according to OS preference
#ifdef _WIN32
  Sleep((int)(1000*duration));
#elif defined(__FreeBSD__) || defined(__linux__) || defined(sgi)
  struct timespec sleep_time, dummy;
  sleep_time.tv_sec = (int)duration;
  sleep_time.tv_nsec = (int)(1000000000*(duration-sleep_time.tv_sec));
  nanosleep(&sleep_time,&dummy);
#endif
}

//----------------------------------------------------------------------------
// Sleep until the specified absolute time has arrived.
// You must pass a handle to the current thread.  
// If '0' is returned, then the thread was aborted before or during the wait.
static int vtkThreadSleep2(vtkMultiThreader::ThreadInfo *data, double time)
{

  vtkPCIBirdWin32TrackedVideoSource *self = (vtkPCIBirdWin32TrackedVideoSource *)(data->UserData);
  for (int i = 0;; i++)
    {
    double remaining = time - vtkTimerLog::GetCurrentTime();

    // check to see if we have reached the specified time
    if (remaining <= 0)
      {
      if (i == 0)
        {
        //vtkGenericWarningMacro("Dropped a video frame.");
        }
      return 1;
      }
    // check the ActiveFlag at least every 0.1 seconds
    if (remaining > 0.1)
      {
      remaining = 0.1;
      }

    // check to see if we are being told to quit 
    data->ActiveFlagLock->Lock();
    int activeFlag = *(data->ActiveFlag);
    data->ActiveFlagLock->Unlock();
    if (activeFlag == 0 || self->GetStopFlag()== 1)
      {
      return 0;
      }

    vtkSleep2(remaining);
    }
}

///Working on adding record capabilities)
//----------------------------------------------------------------------------
// Set the source to grab frames continuously.
// You should override this as appropriate for your device.  

static void *vtkPCIBirdWin32TrackedVideoSourceRecordThread(vtkMultiThreader::ThreadInfo *data)
{
  vtkPCIBirdWin32TrackedVideoSource *self = (vtkPCIBirdWin32TrackedVideoSource *)(data->UserData);
  

  double rate = self->GetFrameRate();
  int maxNumFrames = self->GetMaxNumberOfFrames();
  int frame = 0;
  int port = self->GetRecordingPort();
  double startTime;

  //Variables
  unsigned char *vbptr;
  unsigned char *fptr;
  double *sptr;
  int* framedim = self->GetOutput()->GetDimensions();
  int framesize = framedim[0]*framedim[1]*framedim[2];
  
  // Poll data onces to make sure of the buffers are in place
  self->Poll();
  self->Update();
    
  // Pointer to sensor info
  sptr = self->GetSensorBuffer()->GetPointer(0);  
  cout <<"vtkPCIBirdWin32TrackedVideoSourceRecordThread: Ready to load buffers"<<endl;
  // Start real recording by setting the start time that will fix our sycn clock.
  startTime = vtkTimerLog::GetCurrentTime();
  
  if (!self->GetDoppler()){

  do
    {

    self->Poll();
    self->Update();
    //Put record code here with access members.
    //Copy image information (Convert RGB to Grayscale)
    vbptr = (unsigned char *) self->GetVideoBuffer()->GetScalarPointer(0,0,frame);
    fptr  = (unsigned char *) self->GetOutput()->GetScalarPointer();
        
    for(int i=0;i<framesize;i++) {
    *vbptr=(*fptr + *(fptr+1) + *(fptr+2))/3;
    fptr=fptr+3;
    vbptr++;
    }

    //Copy position information
    *sptr = self->GetSensorMatrix(port)->GetElement(0,3);
    sptr++;
    *sptr = self->GetSensorMatrix(port)->GetElement(1,3);
    sptr++;
    *sptr = self->GetSensorMatrix(port)->GetElement(2,3);
    sptr++;
    *sptr = self->GetProbeBirdAngle(port)->GetValue(0);
    sptr++;
    *sptr = self->GetProbeBirdAngle(port)->GetValue(1);
    sptr++;
    *sptr = self->GetProbeBirdAngle(port)->GetValue(2);
    sptr++;
    frame++;
    if(frame>=maxNumFrames)
      break;
    self->SetProgress((double) frame/maxNumFrames);
    self->InvokeEvent("ProgressEvent");
    }  while (vtkThreadSleep2(data, startTime + frame/rate));
  } else {
   
  do
    {
    self->Poll();
    self->Update();
    //Put record code here with access members.
    //Copy image information (Convert RGB to Grayscale)
    vbptr = (unsigned char *) self->GetVideoBuffer()->GetScalarPointer(0,0,frame);
    fptr  = (unsigned char *) self->GetOutput()->GetScalarPointer();
        
    for(int i=0;i<framesize;i++) {
    *vbptr=*fptr ;
    fptr++;
    vbptr++;
    *vbptr=*fptr;
    fptr++;
    vbptr++;
    *vbptr=*fptr;
    fptr++;
    vbptr++;
    }
    //Copy position information
    *sptr = self->GetSensorMatrix(port)->GetElement(0,3);
    sptr++;
    *sptr = self->GetSensorMatrix(port)->GetElement(1,3);
    sptr++;
    *sptr = self->GetSensorMatrix(port)->GetElement(2,3);
    sptr++;
    *sptr = self->GetProbeBirdAngle(port)->GetValue(0);
    sptr++;
    *sptr = self->GetProbeBirdAngle(port)->GetValue(1);
    sptr++;
    *sptr = self->GetProbeBirdAngle(port)->GetValue(2);
    sptr++;

    frame++;
    if(frame>=maxNumFrames)
      break;
    self->SetProgress((double) frame/maxNumFrames);
    self->InvokeEvent("ProgressEvent");
    }  while (vtkThreadSleep2(data, startTime + frame/rate));
  
  }
  
  //Squeeze data if we have recorded less than maxNumFrames (this should be done out of the thread though)
  self->GetSensorBuffer()->Squeeze();
  self->GetVideoBuffer()->Squeeze();
  return NULL;
}


void vtkPCIBirdWin32TrackedVideoSource::Record()
{
  if (this->Playing)
    {
    this->Stop();
    }

  if (!this->RecordingUS)
    {
    this->Initialize();

    this->RecordingUS = 1;
    this->FrameCount = 0;
    this->Modified();
    //Allocate Arrays
    int region[6];
    this->GetClipRegion(region);
    int framedim[3];
    framedim[0]=region[1]-region[0]+1;
    framedim[1]=region[3]-region[2]+1;
    framedim[2]=region[5]-region[4]+1;
    int framesize = framedim[0]*framedim[1];
    cout <<"Record: Allocating buffer"<<endl;
    this->VideoBuffer->SetDimensions(framedim[0],framedim[1],this->MaxNumberOfFrames);
    if(this->Doppler)
      this->VideoBuffer->SetNumberOfScalarComponents(3);
    else
      this->VideoBuffer->SetNumberOfScalarComponents(1);
    this->VideoBuffer->SetScalarType(VTK_UNSIGNED_CHAR); 
    this->VideoBuffer->UpdateInformation();
    this->VideoBuffer->AllocateScalars();
    this->VideoBuffer->Update();
    this->SensorBuffer->SetNumberOfComponents(6);
    this->SensorBuffer->SetNumberOfTuples(this->MaxNumberOfFrames);
    this->SetProgress(0.0);
  
    //We set recording to zero b/c we do callings on Grab to get continous video.
    this->Recording = 0;

    //Spawn threads
    cout <<"Record: sending threads"<<endl;
    this->PlayerThreadId = 
      this->PlayerThreader->SpawnThread((vtkThreadFunctionType)\
                                        &vtkPCIBirdWin32TrackedVideoSourceRecordThread,this);

    }
}

//----------------------------------------------------------------------------
void vtkPCIBirdWin32TrackedVideoSource::Stop()
{
  if (this->Playing || this->RecordingUS)
    {
    cout<<"Stop: This is the Player Thread ID: "<<this->PlayerThreadId<<endl;
    if (this->PlayerThreadId!=-1) {
    cout<<"Stop: Terminate Thread"<<endl;
    this->PlayerThreader->TerminateThread(this->PlayerThreadId);
    this->PlayerThreadId = -1;
    }
    this->StopFlag = 1;
    this->Playing = 0;
    this->RecordingUS = 0;
    this->Modified();
    }
}

//----------------------------------------------------------------------------
void vtkPCIBirdWin32TrackedVideoSource::CloseConnection()
{
  int l_ErrorCode = 0;
  l_ErrorCode=CloseBIRDSystem();
    
  if(l_ErrorCode != BIRD_ERROR_SUCCESS) {
  vtkErrorMacro("Bird could not be successfully released");
  return;
  }
}

//----------------------------------------------------------------------------
void vtkPCIBirdWin32TrackedVideoSource::UpdateLocatorMatrix(int pSensor)
{
  vtkTransform *l_tempTransform = vtkTransform::New();
  l_tempTransform->PostMultiply();

  //! Tlocator = T tip->ct = Tloc->ct * Tsensor->loc *  Ttip->sensor 
  //! Post multiply --> T' = A*T where T is the current transformation matrix and A is the applied matrix.
  
  //! T = Id
  l_tempTransform->Identity();

  //! T'   = Ttip->sensor * Id
  l_tempTransform->Concatenate(this->m_CalibrationMatrix[pSensor]);
  l_tempTransform->Modified();


  //! T'   = Tsensor->loc  * Ttip->sensor  
  l_tempTransform->Concatenate(this->m_SensorMatrixInLoc[pSensor]);
  l_tempTransform->Modified();
  

  //! T''' = Ttip->ct =  Tloc->ct * Tsensor->loc *  Ttip->sensor   
  l_tempTransform->Concatenate(this->m_RegistrationMatrix);
  l_tempTransform->Modified();


  // set the locator matrix for the specified sensor to the calculated value
  this->m_LocatorMatrix[pSensor]->DeepCopy(l_tempTransform->GetMatrix());

  l_tempTransform->Delete();

}

//----------------------------------------------------------------------------
void vtkPCIBirdWin32TrackedVideoSource::BuildMatrixFromEulerAngles(vtkDoubleArray * angles, vtkMatrix4x4 *matrix)
{
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
//----------------------------------------------------------------------------
void vtkPCIBirdWin32TrackedVideoSource::BuildMatrixFromEulerAngles(double x,double y,double z,double alpha,
                                                                   double beta,double gamma,double mat[4][4])
{

//Rotation
  mat[0][0] = cos(alpha) * cos(beta);
  mat[0][1] = cos(alpha) * sin(beta) * sin(gamma) - sin(alpha) * cos(gamma);
  mat[0][2] = cos(alpha) * sin(beta) * cos(gamma) + sin(alpha) * sin(gamma);
  mat[1][0] = sin(alpha) * cos(beta);
  mat[1][1] = sin(alpha) * sin(beta) * sin(gamma) + cos(alpha) * cos(gamma);
  mat[1][2] = sin(alpha) * sin(beta) *cos(gamma) - cos(alpha) * sin(gamma);
  mat[2][0] = -sin(beta);
  mat[2][1] = cos(beta) * sin(gamma);
  mat[2][2] = cos(beta) * cos(gamma);
//Translation
  mat[0][3] = x;
  mat[1][3] = y;
  mat[2][3] = z;

  mat[3][0] = 0;
  mat[3][1] = 0;
  mat[3][2] = 0;
  mat[3][3] = 1.0;

}

//----------------------------------------------------------------------------
void vtkPCIBirdWin32TrackedVideoSource::BuildEulerAnglesFromMatrix(vtkMatrix4x4 * matrix, vtkDoubleArray *angles)
{
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

//----------------------------------------------------------------------------
void vtkPCIBirdWin32TrackedVideoSource::BuildEulerAnglesFromMatrix(double mat[4][4],double& x, double& y, 
                                                                   double&z, double &alpha, double &beta, double &gamma)
{

  x = mat[0][3];
  y = mat[1][3];
  z = mat[2][3];

  beta = asin(-mat[2][0]);
  alpha = atan2(mat[1][0],mat[0][0]);
  gamma = atan2(mat[2][1],mat[2][2]);

//NORTH POLE
  if (mat[2][0]==-1) {
  beta = PI/2;
  alpha = 0;
  gamma = atan2(mat[0][1],mat[0][2]);
  }
//SOUTH POLE
  if (mat[2][0]==1) {
  beta = -PI/2;
  alpha = 0;
  gamma =  atan2(-mat[1][2],mat[1][1]);
  }

}


//----------------------------------------------------------------------------
void vtkPCIBirdWin32TrackedVideoSource::SetCalibrationMatrixElement(int idPort, int i, int j, double entry)
{
  this->m_CalibrationMatrix[idPort]->SetElement(i,j, entry);

}

//----------------------------------------------------------------------------
void vtkPCIBirdWin32TrackedVideoSource::SetCalibrationMatrix(int idPort, vtkMatrix4x4 *calib_matrix)
{
  this->m_CalibrationMatrix[idPort]->DeepCopy(calib_matrix);  

}

//----------------------------------------------------------------------------
void vtkPCIBirdWin32TrackedVideoSource::SetRegistrationMatrixElement(int i, int j, double entry)
{
  this->m_RegistrationMatrix->SetElement(i,j,entry);

}

//----------------------------------------------------------------------------
void vtkPCIBirdWin32TrackedVideoSource::SetRegistrationMatrix(vtkMatrix4x4 *reg_matrix)
{
  this->m_RegistrationMatrix->DeepCopy(reg_matrix);

}

//----------------------------------------------------------------------------
vtkMatrix4x4 * vtkPCIBirdWin32TrackedVideoSource::GetCalibrationMatrix(int idPort)
{
  return this->m_CalibrationMatrix[idPort];

}

//----------------------------------------------------------------------------
double vtkPCIBirdWin32TrackedVideoSource::GetCalibrationMatrixElement(int idPort, int i, int j)
{
  return this->m_CalibrationMatrix[idPort]->GetElement(i,j);

}

//----------------------------------------------------------------------------
double vtkPCIBirdWin32TrackedVideoSource::GetRegistrationMatrixElement(int i, int j)
{
  return m_RegistrationMatrix->GetElement(i, j);

}

//----------------------------------------------------------------------------
vtkMatrix4x4 *vtkPCIBirdWin32TrackedVideoSource::GetRegistrationMatrix() 
{
  return this->m_RegistrationMatrix;

}

//----------------------------------------------------------------------------
vtkMatrix4x4 * vtkPCIBirdWin32TrackedVideoSource::GetSensorMatrix(int numPort) 
{  
  return this->m_SensorMatrixInLoc[numPort];

}

//----------------------------------------------------------------------------
vtkMatrix4x4 * vtkPCIBirdWin32TrackedVideoSource::GetLocatorMatrix(int numPort)
{
  return this->m_LocatorMatrix[numPort];

}

//----------------------------------------------------------------------------
vtkDoubleArray *vtkPCIBirdWin32TrackedVideoSource::GetProbeBirdAngle(int numPort)
{
  return this->m_ProbeBirdAngles[numPort];

}

int vtkPCIBirdWin32TrackedVideoSource::GetSensorTime(int numPort)
{
  return this->m_SensorTime[numPort];

}
//----------------------------------------------------------------------------
/*void vtkPCIBirdWin32TrackedVideoSource::ErrorMessage(int pErrorCode){

char l_Buffer[100];
int res = GetErrorText(pErrorCode, &l_Buffer[0], sizeof(l_Buffer), SIMPLE_MESSAGE) ;
int l_nBytes = strlen(l_Buffer);
l_Buffer[l_nBytes] = '\n';

fprintf(m_LogFile,"%s",l_Buffer);

}*/


