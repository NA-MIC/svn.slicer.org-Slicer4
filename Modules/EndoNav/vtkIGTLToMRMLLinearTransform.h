/*==========================================================================

  Portions (c) Copyright 2008 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $HeadURL: http://svn.slicer.org/Slicer3/branches/EndoTracking/Modules/OpenIGTLinkIF/vtkIGTLToMRMLLinearTransform.h $
  Date:      $Date: 2009-01-05 13:28:20 -0500 (Mon, 05 Jan 2009) $
  Version:   $Revision: 8267 $

==========================================================================*/

#ifndef __vtkIGTLToMRMLLinearTransform_h
#define __vtkIGTLToMRMLLinearTransform_h

#include "vtkObject.h"
#include "vtkEndoNavWin32Header.h" 
#include "vtkMRMLNode.h"
#include "vtkIGTLToMRMLBase.h"

#include "igtlTransformMessage.h"

class VTK_OPENIGTLINKIF_EXPORT vtkIGTLToMRMLLinearTransform : public vtkIGTLToMRMLBase
{
 public:

  static vtkIGTLToMRMLLinearTransform *New();
  vtkTypeRevisionMacro(vtkIGTLToMRMLLinearTransform,vtkObject);

  void PrintSelf(ostream& os, vtkIndent indent);

  virtual const char*  GetIGTLName() { return "TRANSFORM"; };
  virtual const char*  GetMRMLName() { return "LinearTransform"; };
  virtual vtkIntArray* GetNodeEvents();
  virtual vtkMRMLNode* CreateNewNode(vtkMRMLScene* scene, const char* name);

  //BTX
  virtual int          IGTLToMRML(igtl::MessageBase::Pointer buffer, vtkMRMLNode* node);
  //ETX
  virtual int          MRMLToIGTL(unsigned long event, vtkMRMLNode* mrmlNode, int* size, void** igtlMsg);


 protected:
  vtkIGTLToMRMLLinearTransform();
  ~vtkIGTLToMRMLLinearTransform();

 protected:
  //BTX
  igtl::TransformMessage::Pointer OutTransformMsg;
  //ETX
  
};


#endif //__vtkIGTLToMRMLLinearTransform_h
