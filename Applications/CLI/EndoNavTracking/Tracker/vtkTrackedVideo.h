/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkTrackedVideo.h,v $
  Language:  C++
  Date:      $Date: 2008/06/02 15:55:44 $
  Version:   $Revision: 1.10 $

  =========================================================================*/
// .NAME vtkTrackedVideo
// .SECTION Description
// Combines standardized tracking system and imaging libraries into a 
// tracked, video interface.
//=========================================================================*/

#ifndef __vtkTrackedVideo_h
#define __vtkTrackedVideo_h

//#include "vtkLapUSNavSysConfigure.h"
// Registration.
#include "vtkRegistration.h"
// Abstract base classes.
#include "vtkVideoSource.h"
#include "vtkTrackingSystem.h"
// Other.
#include "vtkImageData.h"
#include "vtkFloatArray.h"
#include "vtkMatrix4x4.h"
#include "vtkMultiThreader.h"
#include "vtkMutexLock.h"
#include "vtkTimerLog.h"
#include "vtkMultiThreader.h"
#include "vtkTrackedVideoData.h"


class vtkTrackedVideo: public vtkProcessObject {
public:

  static vtkTrackedVideo *New();  
  vtkTypeRevisionMacro(vtkTrackedVideo, vtkProcessObject);
  void PrintSelf(ostream &os, vtkIndent indent);

  vtkSetObjectMacro(VideoSource,vtkVideoSource);
  vtkGetObjectMacro(VideoSource,vtkVideoSource);

    vtkSetObjectMacro(TrackingSystem,vtkTrackingSystem);
  vtkGetObjectMacro(TrackingSystem,vtkTrackingSystem);

  void SetRecordBuffer(vtkTrackedVideoData *record_buffer);
  vtkTrackedVideoData *GetRecordBuffer();

  int InitializeTrackingSystem();
    int InitializeVideoSource();
  int Initialize();

  void Record();
  void Stop();
  void Clear();

  void Lock();
  void Unlock();

  vtkSetMacro(MaxNumberOfFrames, int);
  vtkGetMacro(MaxNumberOfFrames, int);
  vtkGetMacro(StopFlag, int);
  vtkGetMacro(RecordingUS, int);
  vtkGetMacro(RecordedSensorIndex, int);
  vtkSetMacro(RecordedSensorIndex, int);
  vtkGetMacro(Doppler,int);
  vtkSetMacro(Doppler,int);
  vtkBooleanMacro(Doppler,int);
  vtkSetMacro(SlicerLoopActive, int);
  vtkGetMacro(SlicerLoopActive, int);
  vtkSetMacro(SaveImages,int);
  vtkGetMacro(SaveImages,int);
  
protected:
  vtkTrackedVideo();
  ~vtkTrackedVideo();

private:
  vtkTrackedVideo(const vtkTrackedVideo&);  // Not implemented.
  void operator=(const vtkTrackedVideo&);  // Not implemented.

  vtkVideoSource *VideoSource;
  vtkTrackingSystem *TrackingSystem;

  int ThreadID;
  vtkMultiThreader *Threader;
  vtkMutexLock *Mutex;

  int RecordingUS;
  int StopFlag;
  int SlicerLoopActive;
  int SaveImages;

  int Doppler;
  int RecordedSensorIndex;
  int MaxNumberOfFrames;

  vtkTrackedVideoData *RecordBuffer;
  vtkTrackedVideoData *LoopBuffer;
};

#endif
