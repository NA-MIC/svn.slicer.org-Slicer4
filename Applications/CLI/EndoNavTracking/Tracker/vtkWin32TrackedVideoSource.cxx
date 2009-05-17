/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkWin32TrackedVideoSource.cxx,v $
  Language:  C++
  Date:      $Date: 2007/07/12 16:24:29 $
  Version:   $Revision: 1.2 $

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
#include <ctype.h>
#include <string.h>
#include <stdlib.h>
#include <cstdlib>
#include "vtkWin32TrackedVideoSource.h"
#include "vtkObjectFactory.h"
#include "vtkMatrix4x4.h"
#include "Bird.h"

using namespace std;

//----------------------------------------------------------------------------
vtkWin32TrackedVideoSource* vtkWin32TrackedVideoSource::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkWin32TrackedVideoSource");
  if(ret)
    {
    return (vtkWin32TrackedVideoSource *)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkWin32TrackedVideoSource;
}

//----------------------------------------------------------------------------
vtkWin32TrackedVideoSource::vtkWin32TrackedVideoSource()
{
  int i;
 
  // see vtkWin32VideoSource for initial values for FrameRate,
  // FlipFrames, FrameBufferRowAlignment, BitMapSize, etc.

  this->BirdInitialized = 0;
  this->ActiveLocator = 1;

  for(i=1;i <= LOCAL_MAX_BIRDS;i++) {
    this->BirdMatrix[i] = vtkMatrix4x4::New();
    this->BirdMatrix[i]->Identity();
    this->m_CalibrationMatrix[i] = vtkMatrix4x4::New();
    this->m_CalibrationMatrix[i]->Identity();
  }
  this->RegistrationMatrix = vtkMatrix4x4::New();
  this->RegistrationMatrix->Identity();
  this->LocatorMatrix = vtkMatrix4x4::New();
  this->LocatorMatrix->Identity();
}

//----------------------------------------------------------------------------
vtkWin32TrackedVideoSource::~vtkWin32TrackedVideoSource()
{
  birdShutDown(1);
  this->BirdInitialized = 0;

  // releases system resources and resets internal 
  // settings
    this->vtkWin32VideoSource::~vtkWin32VideoSource();
}

//----------------------------------------------------------------------------
void vtkWin32TrackedVideoSource::PrintSelf(ostream& os, vtkIndent indent)
{
  vtkVideoSource::PrintSelf(os,indent);
}

//----------------------------------------------------------------------------
int vtkWin32TrackedVideoSource::OpenConnection(int port, int baud, int numbirds, int video)
{
  int i;
  WORD comport[LOCAL_MAX_BIRDS + 1];

  if (this->BirdInitialized) {this->CloseConnection();}

  // Video Initialization
  if (video == 1) {
    this->Initialize();
    this->useVideo = 1;
  } else {
    this->useVideo = 0;
  }

  // Locator Initialization
  for (i=1; i <= LOCAL_MAX_BIRDS; i++) {comport[i] = 0;}
  this->Standalone = !(numbirds);
  comport[(int)(!Standalone)] = 1;

  if (birdRS232WakeUp(1,this->Standalone,numbirds,comport,(DWORD)baud,1000,1000))
  {
    this->NumLocators = numbirds;

    // get the system configuration
    birdGetSystemConfig(1,&this->Sysconfig);

    // get a device configuration for each bird and configure each properly
    for (i=1; i <= this->NumLocators; i++)
    {
      birdGetDeviceConfig(1,i,&this->Devconfig[i]);
      this->Devconfig[i].byDataFormat = BDF_POSITIONMATRIX;
      birdSetDeviceConfig(1,i,&this->Devconfig[i]);
    }

    this->BirdInitialized = 1;
// modif    return(this->NumLocators);
        return(this->BirdInitialized);
  } else {
      return(-1);
  }

}

//----------------------------------------------------------------------------
void vtkWin32TrackedVideoSource::CloseConnection()
{
  this->vtkWin32VideoSource::ReleaseSystemResources();
  birdShutDown(1);
  this->BirdInitialized = 0;
}

//----------------------------------------------------------------------------
void vtkWin32TrackedVideoSource::Poll()
{
  int i, j, k, pos_scale;
  BIRDFRAME birdDataFrame;

  birdStartSingleFrame(1);
  if (this->useVideo == 1) this->vtkWin32VideoSource::Grab();

  birdGetFrame(1,&birdDataFrame);
  if (this->useVideo == 1) this->vtkImageSource::UpdateInformation();

  // update the position and orientation matrix of each sensor 
  for(i=1;0<i && i<=this->NumLocators;i++)
  {
    BIRDREADING *preading;

    if(this->Standalone) {i=0;}

    preading = &birdDataFrame.reading[i];
    pos_scale = this->Devconfig[i].wScaling;

    for(j=0;j<3;j++) {
      for(k=0;k<3;k++) {
        this->BirdMatrix[i]->SetElement(j,k,(double)(preading->matrix.n[k][j] / 32767.0));
      }
    }
    this->BirdMatrix[i]->SetElement(3,0,(double)(preading->position.nX * pos_scale / 32767.0) * INCH_TO_CM);
    this->BirdMatrix[i]->SetElement(3,1,(double)(preading->position.nY * pos_scale / 32767.0) * INCH_TO_CM);
    this->BirdMatrix[i]->SetElement(3,2,(double)(preading->position.nZ * pos_scale / 32767.0) * INCH_TO_CM);
    this->BirdMatrix[i]->SetElement(0,3,(double)0.0);
    this->BirdMatrix[i]->SetElement(1,3,(double)0.0);
    this->BirdMatrix[i]->SetElement(2,3,(double)0.0);
    this->BirdMatrix[i]->SetElement(3,3,(double)1.0);
  

    // update the orientation angles for active sensor
    if(i == ActiveLocator) {
    BirdAngles[0] = preading->angles.nAzimuth;
    BirdAngles[1] = preading->angles.nElevation;
    BirdAngles[2] = preading->angles.nRoll;
    }
  }
  this->UpdateLocatorMatrix(this->ActiveLocator);
}

//----------------------------------------------------------------------------
void vtkWin32TrackedVideoSource::UpdateLocatorMatrix(int num)
{
  int i,j;
  double value;
  vtkTransform *tempTransform;

  tempTransform = vtkTransform::New();
  tempTransform->PostMultiply();

  tempTransform->Identity();
  tempTransform->Concatenate(this->RegistrationMatrix);
  tempTransform->Concatenate(this->BirdMatrix[num]);
  tempTransform->Concatenate(this->m_CalibrationMatrix[num]);

  for(i=0;i<4;i++) {
    for(j=0;j<4;j++) {
      value = (tempTransform->GetMatrix())->GetElement(i,j);
      this->LocatorMatrix->SetElement(i,j,value);
    }
  }

  tempTransform->Delete();
}


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

double vtkWin32TrackedVideoSource::test()
{
  int i;
  WORD comport[LOCAL_MAX_BIRDS + 1];

  comport[1] = 1;
  for (i = 2; i <= LOCAL_MAX_BIRDS; i++)
  {
    comport[i] = 0;
  }

  if (!birdRS232WakeUp(1,FALSE,1,comport,115200,2000,2000))
  {
    return(-1);
  } else {
    return(1);
  }
}
