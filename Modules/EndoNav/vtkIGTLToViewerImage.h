/*==========================================================================

  Portions (c) Copyright 2008 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $HeadURL: http://svn.slicer.org/Slicer3/branches/EndoTracking/Modules/EndoNavIF/vtkIGTLToViewerImage.h $
  Date:      $Date: 2009-03-26 14:33:00 -0400 (Thu, 26 Mar 2009) $
  Version:   $Revision: 8980 $

==========================================================================*/

#ifndef __vtkIGTLToViewerImage_h
#define __vtkIGTLToViewerImage_h

#include "vtkObject.h"
#include "vtkEndoNavWin32Header.h" 
#include "vtkMRMLNode.h"
#include "vtkIGTLToMRMLBase.h"
#include "vtkImageData.h"
#include "vtkImageViewer.h"

#include "vtkSlicerSliceViewer.h"
#include "vtkSlicerSliceGUI.h"

#include "igtlTransformMessage.h"

class vtkMRMLVolumeNode;

class VTK_ENDONAV_EXPORT vtkIGTLToViewerImage : public vtkIGTLToMRMLBase
{
 public:

  static vtkIGTLToViewerImage *New();
  vtkTypeRevisionMacro(vtkIGTLToViewerImage,vtkObject);

  void PrintSelf(ostream& os, vtkIndent indent);

  virtual const char*  GetIGTLName() { return "IMAGE"; };
  virtual const char*  GetMRMLName() { return "Volume"; };
  virtual vtkIntArray* GetNodeEvents();
  virtual vtkMRMLNode* CreateNewNode(vtkMRMLScene* scene, const char* name);

  //BTX
  virtual int          IGTLToMRML(igtl::MessageBase::Pointer buffer, vtkMRMLNode* node);
  //ETX
  virtual int          MRMLToIGTL(unsigned long event, vtkMRMLNode* mrmlNode, int* size, void** igtlMsg);


  vtkGetObjectMacro( SliceViewer, vtkSlicerSliceViewer);
  vtkSetObjectMacro( SliceViewer, vtkSlicerSliceViewer);

  vtkGetObjectMacro( ImageViewerUS, vtkImageViewer);
  vtkSetObjectMacro( ImageViewerUS, vtkImageViewer);

  vtkGetObjectMacro( SliceGUI, vtkSlicerSliceGUI);
  vtkSetObjectMacro( SliceGUI, vtkSlicerSliceGUI);

protected:
  vtkIGTLToViewerImage();
  ~vtkIGTLToViewerImage();

  void CenterImage(vtkMRMLVolumeNode *volumeNode);
  void FitImage();

 protected:
  //BTX
  igtl::TransformMessage::Pointer OutTransformMsg;
  //ETX

  vtkImageViewer* ImageViewerUS;
  
  vtkSlicerSliceViewer *SliceViewer;
  vtkSlicerSliceGUI *SliceGUI;

  vtkImageData* ImageData;


  int NodeCreated;

};


#endif //__vtkIGTLToViewerImage_h
