/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkTrackFileData.h,v $
  Language:  C++
  Date:      $Date: 2008/04/28 22:04:51 $
  Version:   $Revision: 1.10 $

  =========================================================================*/
// .NAME vtkTrackFileData
// .SECTION Description
// Provides data collection, saving, and loading for TrackedVideo
// application.
//=========================================================================*/

#ifndef __vtkTrackFileData_h
#define __vtkTrackFileData_h

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
#include "vtkPoints.h"
#include "vtkProcessObject.h"
#include "vtkGlobFileNames.h"
#include "vtkImageCast.h"
#include "vtkImageLuminance.h"

#include "vtkTrackingSystem.h"

class vtkTrackFileData: public vtkProcessObject {
public:

  static  vtkTrackFileData *New();  
  vtkTypeRevisionMacro(vtkTrackFileData, vtkProcessObject);
  void  PrintSelf(ostream &os, vtkIndent indent);

  int InitilizeRead(const char *filename, const char *pattern);
  
  void InitilizeWrite(const char *dir, const char *prefix);
  
  int ReadStep(int index, 
               double &timeStamp, 
               vtkMatrix4x4 **calibMatrix, 
               vtkMatrix4x4 **regMatrix, 
               vtkMatrix4x4 **sensorMatrix, 
               vtkImageData **image);
  
  int WriteStep(int index, 
               double timeStamp, 
               vtkMatrix4x4 *calibMatrix, 
               vtkMatrix4x4 *regMatrix, 
               vtkMatrix4x4 *sensorMatrix, 
               vtkImageData *image);

protected:
  vtkTrackFileData();
  ~vtkTrackFileData();

private:
  vtkTrackFileData(const vtkTrackFileData&);  // Not implemented.
  void operator=(const vtkTrackFileData&);  // Not implemented.
  
  vtkStructuredPointsWriter *Writer;
  vtkStructuredPointsReader *Reader;
  vtkImageCast *ImageCast;
  vtkImageLuminance *ImageLuminance;
  vtkTrackingSystem *TrackingSystem;

  vtkImageReslice *Reslice;
  
  vtkGlobFileNames *FileNames;
 
  int NumberSourceFiles;

  vtkImageData *SourceImageData;

//BTX
  std::string OutDirectory;
  std::string OutPrefix;

//ETX
};

#endif
