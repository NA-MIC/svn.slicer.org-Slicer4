/*==========================================================================

  Portions (c) Copyright 2008 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $HeadURL: http://svn.slicer.org/Slicer3/branches/EndoTracking/Modules/EndoNavIF/vtkIGTLToViewerTransform.h $
  Date:      $Date: 2009-01-05 13:28:20 -0500 (Mon, 05 Jan 2009) $
  Version:   $Revision: 8267 $

==========================================================================*/

#ifndef __vtkIGTLToViewerTransform_h
#define __vtkIGTLToViewerTransform_h

#include "vtkObject.h"
#include "vtkEndoNavWin32Header.h" 
#include "vtkMRMLNode.h"
#include "vtkIGTLToMRMLBase.h"
#include "vtkSlicerViewerWidget.h"

#include "vtkImageData.h"
#include "vtkImageViewer.h"
#include "vtkMatrix4x4.h"
#include "vtkImageReslice.h"

#include "igtlTransformMessage.h"

class VTK_ENDONAV_EXPORT vtkIGTLToViewerTransform : public vtkIGTLToMRMLBase
{
 public:

  static vtkIGTLToViewerTransform *New();
  vtkTypeRevisionMacro(vtkIGTLToViewerTransform,vtkObject);

  void PrintSelf(ostream& os, vtkIndent indent);

  virtual const char*  GetIGTLName() { return "TRANSFORM"; };
  virtual const char*  GetMRMLName() { return "LinearTransform"; };
  virtual vtkIntArray* GetNodeEvents();
  virtual vtkMRMLNode* CreateNewNode(vtkMRMLScene* scene, const char* name);

  //BTX
  virtual int          IGTLToMRML(igtl::MessageBase::Pointer buffer, vtkMRMLNode* node);
  //ETX
  virtual int          MRMLToIGTL(unsigned long event, vtkMRMLNode* mrmlNode, int* size, void** igtlMsg);

  vtkGetObjectMacro( Viewer, vtkSlicerViewerWidget);
  vtkSetObjectMacro( Viewer, vtkSlicerViewerWidget);

  vtkGetObjectMacro( ImageViewerCT, vtkImageViewer);
  vtkSetObjectMacro( ImageViewerCT, vtkImageViewer);

  vtkGetObjectMacro( SensorMatrix, vtkMatrix4x4);
  vtkSetObjectMacro( SensorMatrix, vtkMatrix4x4);

  vtkGetObjectMacro( CalibrationMatrix, vtkMatrix4x4);
  vtkSetObjectMacro( CalibrationMatrix, vtkMatrix4x4);

  vtkGetObjectMacro( RegistrationMatrix, vtkMatrix4x4);
  vtkSetObjectMacro( RegistrationMatrix, vtkMatrix4x4);

  vtkGetObjectMacro( ImageDataCT, vtkImageData);
  vtkSetObjectMacro( ImageDataCT, vtkImageData);

  //BTX
  void SetSensorTransformNodeName(std::string &name)
    {
    this->SensorTransformNodeName = name;
    };

  void SetRegistrationTransformNodeName(std::string &name)
    {
    this->RegistrationTransformNodeName = name;
    };

  void SetCalibrationTransformNodeName(std::string &name)
    {
    this->CalibrationTransformNodeName = name;
    };

  void SetCTVolumeNodeName(std::string &name)
    {
    this->CTVolumeNodeName = name;
    };

  //ETX

 protected:
  vtkIGTLToViewerTransform();
  ~vtkIGTLToViewerTransform();

  vtkImageViewer* ImageViewerCT;

 protected:
  //BTX
  igtl::TransformMessage::Pointer OutTransformMsg;
  std::string LocatorID;
  std::string GetLocatorActorID(vtkMRMLScene*  scene);

  std::string SensorTransformNodeName;
  std::string RegistrationTransformNodeName;
  std::string CalibrationTransformNodeName;
  std::string CTVolumeNodeName;

  //ETX

  vtkSlicerViewerWidget *Viewer;
  vtkImageReslice       *Reslice;

  int NodeCreated;

  vtkImageData *ImageDataCT;

  vtkMatrix4x4 *SensorMatrix;
  vtkMatrix4x4 *CalibrationMatrix;
  vtkMatrix4x4 *RegistrationMatrix;
  
};


#endif //__vtkIGTLToViewerTransform_h
