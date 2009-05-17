/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkWin32TrackedVideoSource.h,v $
  Language:  C++
  Date:      $Date: 2007/07/12 16:24:29 $
  Version:   $0.1 $

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
// .NAME vtkWin32TrackedVideoSource - Video-for-Windows video digitizer
//                    with Flock of Birds instrument tracking.
// .SECTION Description
// vtkWin32TrackedVideoSource grabs frames or streaming video from a
// Video for Windows compatible device on the Win32 platform.  It also
// captures tracking data from a flock of birds system.


#ifndef __vtkWin32TrackedVideoSource_h
#define __vtkWin32TrackedVideoSource_h

#define LOCAL_MAX_BIRDS 10
#define INCH_TO_CM 25.4

#include "vtkLapUSNavSysConfigure.h"
#include "vtkWin32VideoSource.h"
#include "vtkMatrix4x4.h"
#include "vtkTransform.h"
#include <windows.h>
#include <winuser.h>
#include <vfw.h>
#include <Bird.h>

class VTK_LAPUSNAVSYS_EXPORT vtkWin32TrackedVideoSource : public vtkWin32VideoSource
{
public:

    static vtkWin32TrackedVideoSource *New();

    vtkTypeMacro(vtkWin32TrackedVideoSource,vtkWin32VideoSource);
    void PrintSelf(ostream& os, vtkIndent indent);

    void SetCalibrationMatrix(int num, int i, int j, double entry){this->m_CalibrationMatrix[num]->SetElement(i,j, entry);}
  void SetRegistrationMatrix(int i, int j, double entry){this->RegistrationMatrix->SetElement(i,j,entry);}
  
  void SetCalibrationMatrixIdentity(int num){this->m_CalibrationMatrix[num]->Identity();}
  void SetRegistrationMatrixIdentity(){this->RegistrationMatrix->Identity();}

  vtkMatrix4x4 *GetLocatorMatrix(int num) {UpdateLocatorMatrix(num); return this->LocatorMatrix;}
  vtkMatrix4x4 *GetBirdMatrix(int num) {return this->BirdMatrix[num];}
  
  int OpenConnection(int port, int baud, int numbirds, int video);
  void Poll();
    void CloseConnection();

  void UpdateLocatorMatrix(int num);
  double test();
//  vtkSetObjectMacro(CalibrationMatrix, vtkMatrix4x4);
//  vtkSetObjectMacro(RegistrationMatrix, vtkMatrix4x4);
  
  vtkGetObjectMacro(LocatorMatrix, vtkMatrix4x4);
  
  short GetBirdAngle(int num) {return BirdAngles[num];}

//  void GetLocatorConfig();
//  void SetLocatorConfig();

  vtkGetMacro(BirdInitialized, int);
  vtkGetMacro(NumLocators, int);
  vtkSetMacro(ActiveLocator, int);


protected:
  vtkWin32TrackedVideoSource();
  ~vtkWin32TrackedVideoSource();
  vtkWin32TrackedVideoSource(const vtkWin32TrackedVideoSource&) {};
  void operator=(const vtkWin32TrackedVideoSource&) {};

  bool Standalone;
  int NumLocators;
  int BirdInitialized;
  int ActiveLocator;
  int useVideo;
 
  BIRDSYSTEMCONFIG Sysconfig;
  BIRDDEVICECONFIG Devconfig[LOCAL_MAX_BIRDS + 1];

  // stores calibration matrix for each active sensor
  vtkMatrix4x4 *m_CalibrationMatrix[LOCAL_MAX_BIRDS + 1];

  // stores position and orientation data for each
  // sensor within transmitter frame
  vtkMatrix4x4 *BirdMatrix[LOCAL_MAX_BIRDS + 1];

  // stores registration matrix for conversion from
  // transmitter to slicer frame
  vtkMatrix4x4 *RegistrationMatrix;
  
  // stores position and orientation data for
  // active sensor 
  vtkMatrix4x4 *LocatorMatrix;

  // stores orientation angles
  short BirdAngles[3];

};

#endif
