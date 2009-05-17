/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkTrackedVideoData.h,v $
  Language:  C++
  Date:      $Date: 2008/04/28 22:04:51 $
  Version:   $Revision: 1.10 $

  =========================================================================*/
// .NAME vtkTrackedVideoData
// .SECTION Description
// Provides data collection, saving, and loading for TrackedVideo
// application.
//=========================================================================*/

#ifndef __vtkTrackedVideoData_h
#define __vtkTrackedVideoData_h

//#include "vtkLapUSNavSysConfigure.h"
#include "vtkImageData.h"
#include "vtkImageAppend.h"
#include "vtkMatrix4x4.h"
#include "vtkCollection.h"
#include "vtkStructuredPointsWriter.h"
#include "vtkStructuredPointsReader.h"
#include "vtkFieldData.h"
#include "vtkDoubleArray.h"
#include "vtkImageReslice.h"
#include "vtkJPEGWriter.h"
#include "vtkPNGWriter.h"
#include "vtkImageMapToWindowLevelColors.h"
#include "vtkPoints.h"
#include "vtkProcessObject.h"


class vtkTrackedVideoData: public vtkProcessObject {
public:

  static  vtkTrackedVideoData *New();  
  vtkTypeRevisionMacro(vtkTrackedVideoData, vtkProcessObject);
  void  PrintSelf(ostream &os, vtkIndent indent);

  void  SetRegistration(vtkMatrix4x4 *RegMatrix);
  vtkGetObjectMacro(RegMatrix, vtkMatrix4x4);

  void  SetCalibration(vtkMatrix4x4 *CalibMatrix);
  vtkGetObjectMacro(CalibMatrix, vtkMatrix4x4);

  double  GetTimeStamp(int num);

    void  AddDataElement(double timeStamp, vtkMatrix4x4 *SensorMatrix);
  vtkMatrix4x4 *GetDataElement(int num);
  vtkDoubleArray *GetDataElementsAsArray(char *name);
  vtkPoints *GetDataPositions();
  vtkPoints *GetDataPositionsAfterRegistration();

  void  AddDataElementV(double timeStamp, vtkMatrix4x4 *SensorMatrix, vtkImageData *image);
  int  GetDataElementV(int num, vtkMatrix4x4 *buf, vtkImageData *imbuf);
  vtkImageData *GetImage(int num);

    int  WriteToTxt(char *fname);
  int  WriteToVtk(char *fname);

  int  ReadFromTxt(char *fname);
  int  ReadFromVtk(char *fname);

  int  GetNumElements();
  int  GetNumImages();
  int  GetImageStored();

  void  Reset(void);
  void  Preallocate(int numFrames);

protected:
  vtkTrackedVideoData();
  ~vtkTrackedVideoData();

private:
  vtkTrackedVideoData(const vtkTrackedVideoData&);  // Not implemented.
  void operator=(const vtkTrackedVideoData&);  // Not implemented.

  int  ImageStored;
  int  NumPreallocFrames;
  int  NumPreallocFramesRecorded;
  int  Preallocated;

  vtkDoubleArray  *Times;
  vtkDoubleArray *DataElementAsArray;
  vtkMatrix4x4  *RegMatrix;
  vtkMatrix4x4  *CalibMatrix;
  vtkCollection  *SensorMatrixCollection;
  vtkCollection  *ImageCollection;
  vtkPoints *DataPositions;
  vtkPoints *DataPositionsAfterRegistration;

  vtkImageData  *prealloc_out;

  void preallocInsertSlice(vtkImageData *inData, int slice);

};

#endif
