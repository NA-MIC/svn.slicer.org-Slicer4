/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkWDMVideoSource.h,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkWDMVideoSource - Windows Driver Model video digitizer

#ifndef __vtkWDMVideoSource_h
#define __vtkWDMVideoSource_h

#include "vtkLapUSNavSysConfigure.h"
#include "vtkVideoSource.h"
#include "vtkCriticalSection.h"
#include "vtkTimerLog.h"
#include "vtkUnsignedCharArray.h"
#include "vtkWin32OpenGLRenderWindow.h"
#include "Dshow.h"
#include "Qedit.h"

class VTK_LAPUSNAVSYS_EXPORT vtkWDMVideoSource : public vtkVideoSource
{
public:
  static vtkWDMVideoSource *New();
  vtkTypeRevisionMacro(vtkWDMVideoSource,vtkVideoSource);
  void PrintSelf(ostream& os, vtkIndent indent);   

  void Initialize();
  void ReleaseSystemResources();
  void InternalGrab();

  void SetFrameSize(int x, int y, int z);
  void SetFrameRate(float rate);
  void SetOutputFormat(int format);

  void VideoFormatDialog();
  void VideoSourceDialog();
  void VideoCrossbarDialog();

protected:
  vtkWDMVideoSource();
  ~vtkWDMVideoSource();

  HRESULT CreateCaptureFilter();
  HRESULT CreateSampleGrabber();
  HRESULT RenderGraph();

    IGraphBuilder *pGraph;
    ICaptureGraphBuilder2 *pBuild;
  ICreateDevEnum *pDevEnum;
  IEnumMoniker *pEnum;
  IMoniker *pMoniker;
  IBaseFilter *pCapF;
  IBaseFilter *pGrabberF;
  ISampleGrabber *pGrabber;
  IBaseFilter *pRenderF;
  IMediaControl *pControl;

private:
  vtkWDMVideoSource(const vtkWDMVideoSource&);  // Not implemented.
  void operator=(const vtkWDMVideoSource&);  // Not implemented.
};



#endif





