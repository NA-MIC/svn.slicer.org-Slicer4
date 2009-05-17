/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkWDMVideoSource.cxx,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkWDMVideoSource.h"

#include "vtkObjectFactory.h"

#include <ctype.h>
#include <windows.h>
#include <winuser.h>
#include <atlbase.h>
#include <atlconv.h>

// VFW compressed formats are listed at http://www.webartz.com/fourcc/
#define VTK_BI_UYVY 0x59565955

vtkCxxRevisionMacro(vtkWDMVideoSource, "$Revision: 1.1 $");
vtkStandardNewMacro(vtkWDMVideoSource);


//----------------------------------------------------------------------------
vtkWDMVideoSource::vtkWDMVideoSource()
{
  HRESULT hr;

  hr = CoInitialize(NULL);

  this->pGraph  = NULL;
    this->pBuild  = NULL;
  this->pDevEnum  = NULL;
  this->pEnum    = NULL;
  this->pMoniker  = NULL;
  this->pCapF    = NULL;
  this->pGrabberF  = NULL;
  this->pGrabber  = NULL;
  this->pRenderF  = NULL;
  this->pControl  = NULL;

  this->Initialized = 0;
  this->FrameRate = 30;
  this->OutputFormat = VTK_RGB;
  this->NumberOfScalarComponents = 3;
  this->FrameBufferBitsPerPixel = 24;
  this->FlipFrames = 0;
  this->FrameBufferRowAlignment = 4;

}

//----------------------------------------------------------------------------
vtkWDMVideoSource::~vtkWDMVideoSource()
{
  this->ReleaseSystemResources();
  CoUninitialize();
}

//----------------------------------------------------------------------------
void vtkWDMVideoSource::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}


//----------------------------------------------------------------------------
void vtkWDMVideoSource::Initialize()
{
  HRESULT hr;

  if (this->Initialized) {
    return;
  }
  
  this->UpdateFrameBuffer();

  //-------------------
    // Capture Graph Builder.
  hr = CoCreateInstance(CLSID_CaptureGraphBuilder2, NULL, CLSCTX_INPROC_SERVER, IID_ICaptureGraphBuilder2, (void**)&(this->pBuild) );
  if (SUCCEEDED(hr)) {
    // Create the Filter Graph Manager.
    hr = CoCreateInstance(CLSID_FilterGraph, 0, CLSCTX_INPROC_SERVER, IID_IGraphBuilder, (void**)&(this->pGraph) );
    if (SUCCEEDED(hr)) {
      hr = this->pBuild->SetFiltergraph(this->pGraph);
      hr = pGraph->QueryInterface(IID_IMediaControl, (void **)&(this->pControl) );
    } else {
      vtkErrorMacro(<< "Initialize: failed graph manager");
      return;
    }
  } else {
    vtkErrorMacro(<< "Initialize: failed graph builder");
    return;
  }

  hr = this->CreateCaptureFilter();
  if (FAILED(hr)) {
    return;
  }

  hr = this->CreateSampleGrabber();
  if (FAILED(hr)) {
    return;
  }

  //-------------------
  // Connect filters (frame grabber -> sample grabber).
  hr = this->RenderGraph();
  if (FAILED(hr)) {
    vtkErrorMacro(<< "Initialize: failed connect filters");
    return;
  }

  //-------------------
  hr = this->pControl->Run();
  
  this->FrameBufferMutex->Lock();
  this->UpdateFrameBuffer();
  this->FrameBufferMutex->Unlock();

  this->Initialized = 1;
}

HRESULT vtkWDMVideoSource::CreateCaptureFilter() {
  
  HRESULT hr;

  // System Device Enumerator.
  if (this->pDevEnum) this->pDevEnum->Release();
  hr = CoCreateInstance(CLSID_SystemDeviceEnum, NULL, CLSCTX_INPROC_SERVER, IID_ICreateDevEnum, (void**)&(this->pDevEnum) );
  if (SUCCEEDED(hr)) {
    pDevEnum->CreateClassEnumerator(CLSID_VideoInputDeviceCategory, &(this->pEnum), 0);
    pEnum->Next(1, &pMoniker, NULL);
    IPropertyBag *pPropBag;
    hr = pMoniker->BindToStorage(0, 0, IID_IPropertyBag, (void**)&pPropBag);
    if (FAILED(hr)) {
      vtkErrorMacro(<< "Initialize: failed bind to capture device");
      return hr;
    }
    /*
    // Find the description or friendly name.
    VARIANT varName;
    VariantInit(&varName);
    hr = pPropBag->Read(L"FriendlyName", &varName, 0);
    vtkErrorMacro(<< "Initialize: enumerate devices: "<< COLE2T(V_BSTR(&varName)));
    VariantClear(&varName);
    pPropBag->Release();
    */
  } else {
    vtkErrorMacro(<< "Initialize: failed device enumerator");
    return hr;
  }
  // Get Source (frame grabber card)
  hr = this->pMoniker->BindToObject(0, 0, IID_IBaseFilter, (void**)&(this->pCapF) );
  if (FAILED(hr)) {
    vtkErrorMacro(<< "Initialize: failed frame grabber");
    return hr;
  }

  hr = this->pGraph->RemoveFilter(this->pCapF);
  hr = this->pGraph->AddFilter(this->pCapF, L"Capture Filter");
  if (FAILED(hr)) {
    vtkErrorMacro(<< "Initialize: failed add frame grabber to graph");
    return hr;
  }

  return hr;
}

HRESULT vtkWDMVideoSource::CreateSampleGrabber() {

  HRESULT hr;

  if (this->pGrabberF) this->pGrabberF->Release();
  hr = CoCreateInstance(CLSID_SampleGrabber, NULL, CLSCTX_INPROC_SERVER, IID_IBaseFilter, (void**)&(this->pGrabberF) );
  if (FAILED(hr)) {
    vtkErrorMacro(<< "Initialize: failed sample grabber filter");
    return hr;
  }
  // Query the ISampleGrabber interface.
  hr = this->pGrabberF->QueryInterface(IID_ISampleGrabber, (void**)&(this->pGrabber) );
  if (FAILED(hr)) {
    vtkErrorMacro(<< "Initialize: failed sample grabber interface");
    return hr;
  }
  
  // Set recording type (buffer or callback)
  this->pGrabber->SetBufferSamples(TRUE);
  this->pGrabber->SetOneShot(FALSE);
  // Set initial input media type (video) and subtype (i.e. rgb24).
  AM_MEDIA_TYPE mt;
  ZeroMemory(&mt, sizeof(AM_MEDIA_TYPE));
  mt.majortype = MEDIATYPE_Video;
  mt.subtype = MEDIASUBTYPE_RGB24;
  this->pGrabber->SetMediaType(&mt);

  hr = this->pGraph->RemoveFilter(this->pGrabberF);
  hr = this->pGraph->AddFilter(this->pGrabberF, L"Sample Grabber");
  if (FAILED(hr)) {
    vtkErrorMacro(<< "Initialize: failed add sample grabber to graph");
    return hr;
  }

  return hr;
}


HRESULT vtkWDMVideoSource::RenderGraph() {

  return this->pBuild->RenderStream(&PIN_CATEGORY_CAPTURE, &MEDIATYPE_Video, this->pCapF, NULL, this->pGrabberF);
}


//----------------------------------------------------------------------------
void vtkWDMVideoSource::ReleaseSystemResources()
{
  if (this->pControl) {
    this->pControl->Stop();
    this->pControl->Release();
  }

  if (this->pGraph)
    this->pGraph->Release();
    if (this->pBuild)
    this->pBuild->Release();
  if (this->pDevEnum)
    this->pDevEnum->Release();
  if (this->pEnum)
    this->pEnum->Release();
  if (this->pMoniker)
    this->pMoniker->Release();
  if (this->pCapF)
    this->pCapF->Release();
  if (this->pGrabberF)
    this->pGrabberF->Release();
  if (this->pGrabber)
    this->pGrabber->Release();  

  this->Initialized = 0;
}

//----------------------------------------------------------------------------
// copy the Device Independent Bitmap from the capture framebuffer into the
// vtkVideoSource framebuffer (don't do the unpacking yet)
void vtkWDMVideoSource::InternalGrab()
{
  HRESULT hr;

  if (!this->Initialized) {
    return;
  }

  // get a thread lock on the frame buffer
  this->FrameBufferMutex->Lock();
 
  if (this->AutoAdvance) {
    this->AdvanceFrameBuffer(1);
    if (this->FrameIndex + 1 < this->FrameBufferSize) {
      this->FrameIndex++;
    }
  }
  int index = this->FrameBufferIndex;
  // record time stamp
  this->FrameBufferTimeStamps[index] = vtkTimerLog::GetCurrentTime();
  if (this->FrameCount++ == 0) {
      this->StartTimeStamp = this->FrameBufferTimeStamps[index];
  }

  // copy image
  long cbBuffer = 0;
  hr = this->pGrabber->GetCurrentBuffer(&cbBuffer, NULL);
  if (FAILED(hr)) {
    TCHAR szErr[MAX_ERROR_TEXT_LEN];
        DWORD res = AMGetErrorText(hr, szErr, MAX_ERROR_TEXT_LEN);
    vtkErrorMacro(<< "Grab: failed buffer length \n" << szErr);
  } else {
    unsigned char *pBuffer = new unsigned char[cbBuffer];
    hr = this->pGrabber->GetCurrentBuffer(&cbBuffer, (long*)pBuffer);
    unsigned char *cptrDIB = pBuffer;
    if (FAILED(hr)) {
      TCHAR szErr[MAX_ERROR_TEXT_LEN];
      DWORD res = AMGetErrorText(hr, szErr, MAX_ERROR_TEXT_LEN);
      vtkErrorMacro(<< "Grab: failed get buffer \n" << szErr);
    } else {
      IAMStreamConfig *pConfig;
      AM_MEDIA_TYPE *pmtConfig;
      this->pBuild->FindInterface(&PIN_CATEGORY_CAPTURE, &MEDIATYPE_Video, this->pCapF, IID_IAMStreamConfig, (void**)&pConfig);
      pConfig->GetFormat(&pmtConfig);
      unsigned char *ptr = (unsigned char *)((reinterpret_cast<vtkUnsignedCharArray*>(this->FrameBuffer[index]))->GetPointer(0));

      // the DIB has rows which are multiples of 4 bytes
      int outBytesPerRow = ((this->FrameBufferExtent[1]-this->FrameBufferExtent[0]+1) * this->FrameBufferBitsPerPixel + 7)/8;
      outBytesPerRow += outBytesPerRow % this->FrameBufferRowAlignment;
      int inBytesPerRow = this->FrameSize[0] * (HEADER(pmtConfig->pbFormat)->biBitCount/8);
      outBytesPerRow += outBytesPerRow % 4;
      int rows = this->FrameBufferExtent[3]-this->FrameBufferExtent[2]+1;

      cptrDIB += this->FrameBufferExtent[0]*(HEADER(pmtConfig->pbFormat)->biBitCount/8);
      cptrDIB += this->FrameBufferExtent[2]*inBytesPerRow;

      if (outBytesPerRow == inBytesPerRow) {
        memcpy(ptr,cptrDIB,inBytesPerRow*rows);
      } else {
        while (--rows >= 0) {
          memcpy(ptr,cptrDIB,outBytesPerRow);
          ptr += outBytesPerRow;
          cptrDIB += inBytesPerRow;
        }
      }

      pConfig->Release();
    }
    free(pBuffer);
  }

  this->Modified();

  this->FrameBufferMutex->Unlock();
}


//----------------------------------------------------------------------------
// codecs

static inline void vtkYUVToRGB(unsigned char *yuv, unsigned char *rgb)
{ 
  /* // floating point 
  int Y = yuv[0] - 16;
  int U = yuv[1] - 128;
  int V = yuv[2] - 128;

  int R = 1.164*Y + 1.596*V           + 0.5;
  int G = 1.164*Y - 0.813*V - 0.391*U + 0.5;
  int B = 1.164*Y           + 2.018*U + 0.5;
  */

  // integer math
  int Y = (yuv[0] - 16)*76284;
  int U = yuv[1] - 128;
  int V = yuv[2] - 128;

  int R = Y + 104595*V           ;
  int G = Y -  53281*V -  25625*U;
  int B = Y            + 132252*U;

  // round
  R += 32768;
  G += 32768;
  B += 32768;

  // shift
  R >>= 16;
  G >>= 16;
  B >>= 16;

  // clamp
  if (R < 0) { R = 0; }
  if (G < 0) { G = 0; }
  if (B < 0) { B = 0; }

  if (R > 255) { R = 255; };
  if (G > 255) { G = 255; };
  if (B > 255) { B = 255; };

  // output
  rgb[0] = R;
  rgb[1] = G;
  rgb[2] = B;
}

//----------------------------------------------------------------------------
void vtkWDMVideoSource::VideoFormatDialog()
{
  HRESULT hr;
  HWND hwndParent;
  vtkWin32OpenGLRenderWindow *rw = vtkWin32OpenGLRenderWindow::New();

  this->Initialize();
  if (!this->Initialized) {
    return;
    }

  this->pControl->Stop(); // Stop the graph.
  this->pGraph->RemoveFilter(this->pGrabberF);

  IAMStreamConfig *pConfig;
  ISpecifyPropertyPages *pSpec;
  CAUUID cauuid;
  AM_MEDIA_TYPE *pmtConfig;

  hr = this->pBuild->FindInterface(&PIN_CATEGORY_CAPTURE, &MEDIATYPE_Video, this->pCapF, IID_IAMStreamConfig, (void**)&pConfig);
  if (FAILED(hr)) goto end;
  hr = pConfig->QueryInterface(IID_ISpecifyPropertyPages, (void**)&pSpec);
  if (FAILED(hr)) goto end;
  hwndParent = rw->GetWindowId();
  pSpec->GetPages(&cauuid);
  OleCreatePropertyFrame(hwndParent, 30, 30, NULL, 1, (IUnknown **)&pConfig, cauuid.cElems, (GUID *)cauuid.pElems, 0, 0, NULL);

  pConfig->GetFormat(&pmtConfig);
  //vtkErrorMacro(<< "SetFrameSize: " << HEADER(pmtConfig->pbFormat)->biWidth << " x " << HEADER(pmtConfig->pbFormat)->biHeight );
  this->FrameSize[0] = HEADER(pmtConfig->pbFormat)->biWidth;
  this->FrameSize[1] = HEADER(pmtConfig->pbFormat)->biHeight;
  this->FrameSize[2] = 1;
  this->Modified();

  if (this->Initialized) {
    this->FrameBufferMutex->Lock();
    this->UpdateFrameBuffer();
    this->FrameBufferMutex->Unlock();
    }

  CoTaskMemFree(cauuid.pElems);
  pSpec->Release();
  pConfig->Release();

end:
  this->CreateSampleGrabber();
  this->RenderGraph();
  this->pControl->Run();

  rw->Delete();
}

//----------------------------------------------------------------------------
void vtkWDMVideoSource::VideoSourceDialog()
{
  HRESULT hr;
  HWND hwndParent;
  vtkWin32OpenGLRenderWindow *rw = vtkWin32OpenGLRenderWindow::New();

  this->Initialize();
  if (!this->Initialized) {
    return;
    }

  this->pControl->Stop(); // Stop the graph.
  this->pGraph->RemoveFilter(this->pGrabberF);

  ISpecifyPropertyPages *pSpec;
  CAUUID cauuid;

  hr = this->pCapF->QueryInterface(IID_ISpecifyPropertyPages, (void**)&pSpec);
  if (FAILED(hr)) goto end;
  hwndParent = rw->GetWindowId();
  pSpec->GetPages(&cauuid);
  OleCreatePropertyFrame(hwndParent, 30, 30, NULL, 1, (IUnknown **)&pCapF, cauuid.cElems, (GUID *)cauuid.pElems, 0, 0, NULL);
  CoTaskMemFree(cauuid.pElems);
  pSpec->Release();

end:
  this->CreateSampleGrabber();
  this->RenderGraph();
  this->pControl->Run();

  rw->Delete();
}


//----------------------------------------------------------------------------
void vtkWDMVideoSource::VideoCrossbarDialog()
{
  HRESULT hr;
  HWND hwndParent;
  vtkWin32OpenGLRenderWindow *rw = vtkWin32OpenGLRenderWindow::New();

  this->Initialize();
  if (!this->Initialized) {
    return;
    }

  this->pControl->Stop(); // Stop the graph.
  this->pGraph->RemoveFilter(this->pGrabberF);

  IAMCrossbar *pCrossbar;
  ISpecifyPropertyPages *pSpec;
  CAUUID cauuid;

  hr = this->pBuild->FindInterface(&PIN_CATEGORY_CAPTURE, &MEDIATYPE_Video, this->pCapF, IID_IAMCrossbar, (void**)&pCrossbar);
  if (FAILED(hr)) goto end;
  hr = pCrossbar->QueryInterface(IID_ISpecifyPropertyPages, (void**)&pSpec);
  if (FAILED(hr)) goto end;
  hwndParent = rw->GetWindowId();
  pSpec->GetPages(&cauuid);
  OleCreatePropertyFrame(hwndParent, 30, 30, NULL, 1, (IUnknown **)&pCrossbar, cauuid.cElems, (GUID *)cauuid.pElems, 0, 0, NULL);
  CoTaskMemFree(cauuid.pElems);
  pCrossbar->Release();
  pSpec->Release();

end:
  this->CreateSampleGrabber();
  this->RenderGraph();
  this->pControl->Run();

  rw->Delete();
}


//----------------------------------------------------------------------------
// try for the specified frame size
void vtkWDMVideoSource::SetFrameSize(int x, int y, int z)
{
  HRESULT hr;

  if (x == this->FrameSize[0] && y == this->FrameSize[1] && z == this->FrameSize[2]) {
    return;
    }

  if (x < 1 || y < 1 || z != 1) {
      vtkErrorMacro(<< "SetFrameSize: Illegal frame size");
    return;
    }

  this->pControl->Stop();
  this->pGraph->RemoveFilter(this->pGrabberF);

  IAMStreamConfig *pConfig = NULL;
  int iCount = 0, iSize = 0;
  VIDEO_STREAM_CONFIG_CAPS scc;
  AM_MEDIA_TYPE *pmtConfig;

  this->pBuild->FindInterface(&PIN_CATEGORY_CAPTURE, &MEDIATYPE_Video, this->pCapF, IID_IAMStreamConfig, (void **)&pConfig);
  pConfig->GetNumberOfCapabilities(&iCount, &iSize);

  if (iSize == sizeof(VIDEO_STREAM_CONFIG_CAPS)) {
    for (int iFormat = 0; iFormat < iCount; iFormat++) {
      hr = pConfig->GetStreamCaps(iFormat, &pmtConfig, (BYTE*)&scc);
      if (SUCCEEDED(hr)) {
        /* Examine the format, and possibly use it. */
        if ((pmtConfig->majortype == MEDIATYPE_Video) &&
          (pmtConfig->subtype == MEDIASUBTYPE_RGB24) &&
          (pmtConfig->formattype == FORMAT_VideoInfo) &&
          (pmtConfig->cbFormat >= sizeof (VIDEOINFOHEADER)) &&
          (pmtConfig->pbFormat != NULL)) {

          VIDEOINFOHEADER *pVih = (VIDEOINFOHEADER*)pmtConfig->pbFormat;
          pVih->bmiHeader.biWidth = x;    
          pVih->bmiHeader.biHeight = y;
          pVih->bmiHeader.biSizeImage = DIBSIZE(pVih->bmiHeader);
          hr = pConfig->SetFormat(pmtConfig);
          if (FAILED(hr)) {
            TCHAR szErr[MAX_ERROR_TEXT_LEN];
            DWORD res = AMGetErrorText(hr, szErr, MAX_ERROR_TEXT_LEN);
            vtkErrorMacro(<< "SetFrameSize failed: \n" << szErr);
          }

        }
      }
    }
  }

  this->CreateSampleGrabber();
  this->RenderGraph();
  this->pControl->Run();

  pConfig->GetFormat(&pmtConfig);
  //vtkErrorMacro(<< "SetFrameSize: " << HEADER(pmtConfig->pbFormat)->biWidth << " x " << HEADER(pmtConfig->pbFormat)->biHeight );
  this->FrameSize[0] = HEADER(pmtConfig->pbFormat)->biWidth;
  this->FrameSize[1] = HEADER(pmtConfig->pbFormat)->biHeight;
  this->FrameSize[2] = z;
  this->Modified();

  if (this->Initialized) {
    this->FrameBufferMutex->Lock();
    this->UpdateFrameBuffer();
    this->FrameBufferMutex->Unlock();
    }

  pConfig->Release();
}

//----------------------------------------------------------------------------
void vtkWDMVideoSource::SetFrameRate(float rate)
{
  if (rate == this->FrameRate) {
    return;
    }

  if (this->Initialized) {
//    this->FrameRate = rate;
//    this->Modified();
//    set frame rate
  }
}

//----------------------------------------------------------------------------
void vtkWDMVideoSource::SetOutputFormat(int format)
{
  if (format == this->OutputFormat) {
    return;
    }

  this->OutputFormat = format;

  // convert color format to number of scalar components
  int numComponents;

  switch (this->OutputFormat) {
  case VTK_RGBA:
      numComponents = 4;
      break;
    case VTK_RGB:
      numComponents = 3;
      break;
    case VTK_LUMINANCE:
      numComponents = 1;
      break;
    default:
      numComponents = 0;
      vtkErrorMacro(<< "SetOutputFormat: Unrecognized color format.");
      break;
    }
  this->NumberOfScalarComponents = numComponents;

  if (this->FrameBufferBitsPerPixel != numComponents*8) {
    this->FrameBufferMutex->Lock();
    this->FrameBufferBitsPerPixel = numComponents*8;
    if (this->Initialized) {
      this->UpdateFrameBuffer();
    }
    this->FrameBufferMutex->Unlock();
    }

  this->Modified();
}
