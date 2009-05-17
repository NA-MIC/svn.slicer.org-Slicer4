/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkTrackedVideo.cxx,v $
  Language:  C++
  Date:      $Date: 2008/06/02 15:55:44 $
  Version:   $Revision: 1.16 $

  =========================================================================*/
#include "vtkTrackedVideo.h"
#include <iostream>
#include "windows.h"

using namespace std;

vtkCxxRevisionMacro(vtkTrackedVideo, "$Revision: 1.16 $");
vtkStandardNewMacro(vtkTrackedVideo);

vtkTrackedVideo::vtkTrackedVideo() {

  // Initialize video soure.
  this->VideoSource = NULL;

  // Initialize tracking system.
  this->TrackingSystem = NULL;

  // Initialize recording variables.
  this->Threader = vtkMultiThreader::New();
  this->ThreadID = -1;
  this->Mutex = vtkMutexLock::New();
  this->MaxNumberOfFrames = 1;
  this->RecordedSensorIndex = 0;
  this->RecordBuffer = vtkTrackedVideoData::New();
  this->LoopBuffer = vtkTrackedVideoData::New();
  this->RecordingUS = FALSE;
  this->StopFlag = FALSE;
  this->Doppler = FALSE;
  this->SlicerLoopActive = 0;
  this->SaveImages = 0;

}

vtkTrackedVideo::~vtkTrackedVideo() {

  // Delete video source.
  this->SetVideoSource(NULL);

  // Delete tracking system.
  this->SetTrackingSystem(NULL);

}

void vtkTrackedVideo::PrintSelf(ostream &os, vtkIndent indent) {

  // Video source information.
  cout << "Video source information: " << endl;
  cout << "--------------------------" << endl;
  if(this->VideoSource != NULL) {
    this->VideoSource->PrintSelf(os, indent);
  } else {
    cout << "** Video source not set." << endl;
  }
  cout << "\n\n";
  
  // Tracking system information.
  cout << "Tracking system information: " << endl;
  cout << "-----------------------------" << endl;
  if(this->TrackingSystem != NULL) {
    this->TrackingSystem->PrintSelf(os, indent);
  } else {
    cout << "** Tracking system not set." << endl;
  }
  cout << "\n";

  // Recording status variables.
  cout << "Recording Information: " << endl;
  cout << "------------------" << endl;
  cout << "Recording status: " << this->RecordingUS << endl;
  cout << "Stop flag status: " << this->StopFlag << endl;
  cout << "Doppler status: " << this->Doppler << endl;
  cout << "Recorded sensor index: " << this->RecordedSensorIndex << endl;
  cout << "Max number of recorded frames: " << this->MaxNumberOfFrames << endl;

}

void vtkTrackedVideo::SetRecordBuffer(vtkTrackedVideoData *record_buffer) {
  if (this->RecordBuffer) {
    this->RecordBuffer->Delete();
  }
  this->RecordBuffer = record_buffer;
}

vtkTrackedVideoData *vtkTrackedVideo::GetRecordBuffer() {

  return this->RecordBuffer;
}

int vtkTrackedVideo::InitializeTrackingSystem() {

  // Check that tracking system is initialized.
  if(this->TrackingSystem == NULL) 
    return false;

  // Attempt to establish connection.
  return this->TrackingSystem->OpenConnection();
}

int vtkTrackedVideo::InitializeVideoSource() {
  
  // Check that tracking system is initialized.
  if(this->VideoSource == NULL) 
    return false;

  // Attempt to establish connection.
  this->VideoSource->Initialize();
  if(!this->VideoSource->GetInitialized()) return false;
 
  return true;
}

int vtkTrackedVideo::Initialize() {

  if (this->InitializeTrackingSystem() == false) {
    vtkWarningMacro("Tracking System could not be initialized. Maybe Tracking class is missing");
      return false;
  }
  if (this->InitializeVideoSource() == false) {
    vtkWarningMacro("Video Source could not be initialized. Maybe Video class is missing");
        return false;
  }
  this->RecordBuffer->Reset();

    return true;
}

// Platform-independent sleep function.
static inline void vtkSleep(double duration) {

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

// Set the source to grab frames continuously.
// You should override this as appropriate for your device. 
static void *vtkTrackedVideoRecordThread(vtkMultiThreader::ThreadInfo *data) {

  vtkTrackedVideo *self = (vtkTrackedVideo *)(data->UserData);
  vtkMatrix4x4 *mat;
  vtkImageData *image;
  vtkTrackedVideoData *recbuf = self->GetRecordBuffer();
  
  double startTime, currTime, remaining;
  double rate = self->GetVideoSource()->GetFrameRate();

  // Update video source once to make sure buffers are in place.
  self->GetVideoSource()->Update();

  // Start real recording by setting the start time that will fix our sycn clock.
  startTime = vtkTimerLog::GetCurrentTime();

  if (self->GetSaveImages() == 1) {
    recbuf->Preallocate(self->GetMaxNumberOfFrames());
  } else {
    recbuf->Preallocate(0);
  }
  recbuf->Preallocate(0);

  while (!self->GetStopFlag() && (recbuf->GetNumElements() < self->GetMaxNumberOfFrames())) {

    self->GetTrackingSystem()->UpdateLocatorMatrices();
    self->GetVideoSource()->Grab();

    currTime = vtkTimerLog::GetCurrentTime();
    mat = self->GetTrackingSystem()->GetSensorMatrix(self->GetRecordedSensorIndex());
    image = self->GetVideoSource()->GetOutput();
    image->Update();

    if (self->GetSaveImages() == 1) {
      recbuf->AddDataElementV(currTime,mat,image);
    } else {
      recbuf->AddDataElement(currTime,mat);
    }

    self->SetProgress((double)recbuf->GetNumElements()/(double)self->GetMaxNumberOfFrames());
    if (self->GetSlicerLoopActive() == 0)
      self->InvokeEvent("ProgressEvent");

    // sleep between frames
    remaining = startTime + (double)recbuf->GetNumElements()/rate - vtkTimerLog::GetCurrentTime();
    if (remaining <= 0) {
      vtkGenericWarningMacro("Dropped a video frame");
      remaining = 0;
    } else if (remaining > 0.1) {
      remaining = 0.1;
    }
    vtkSleep(remaining);

  }
  self->Stop();

  return NULL;
}

void vtkTrackedVideo::Record() {
 
  // If the video source is already feeding data, stop the feed.
  if(this->VideoSource->GetPlaying()) 
    this->VideoSource->Stop();

  /* Initialize the recording process. */
  if(!this->RecordingUS) {
    this->RecordingUS = TRUE;
    this->StopFlag = FALSE;

    this->VideoSource->Initialize();
    this->VideoSource->SetFrameCount(0);
    this->VideoSource->Modified();

    this->SetProgress(0.0);
    this->RecordBuffer->Reset();
  
    // We set recording to zero b/c we do callings on Grab() to get continous video.
    //this->VideoSource->Recording = 0;

    // Spawn threads.
    this->ThreadID = this->Threader->SpawnThread((vtkThreadFunctionType) &vtkTrackedVideoRecordThread,this);
  }
}

void vtkTrackedVideo::Stop() {

  if(this->RecordingUS) {
    this->StopFlag = TRUE;
        this->RecordingUS = FALSE;
  }

}

void vtkTrackedVideo::Clear() {

  if(this->RecordingUS) {
    this->Stop();
  }
  this->SetProgress(0.0);
  this->RecordBuffer->Reset();
}

void vtkTrackedVideo::Lock() {
  this->Mutex->Lock();
}

void vtkTrackedVideo::Unlock() {
  this->Mutex->Unlock();
}
