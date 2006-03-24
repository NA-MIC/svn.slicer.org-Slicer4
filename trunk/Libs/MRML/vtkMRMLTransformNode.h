/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkMRMLTransformNode.h,v $
  Date:      $Date: 2006/03/19 17:12:29 $
  Version:   $Revision: 1.13 $

=========================================================================auto=*/
// .NAME vtkMRMLTransformNode - MRML node for representing a volume (image stack).
// .SECTION Description
// Volume nodes describe data sets that can be thought of as stacks of 2D 
// images that form a 3D volume.  Volume nodes describe where the images 
// are stored on disk, how to render the data (window and level), and how 
// to read the files.  This information is extracted from the image 
// headers (if they exist) at the time the MRML file is generated.  
// Consequently, MRML files isolate MRML browsers from understanding how 
// to read the myriad of file formats for medical data. 

#ifndef __vtkMRMLTransformNode_h
#define __vtkMRMLTransformNode_h

#include "vtkMRML.h"
#include "vtkMRMLNode.h"

#include "vtkGeneralTransform.h"
#include "vtkMatrix4x4.h"

class vtkImageData;

class VTK_MRML_EXPORT vtkMRMLTransformNode : public vtkMRMLNode
{
  public:
  static vtkMRMLTransformNode *New(){return NULL;};
  vtkTypeMacro(vtkMRMLTransformNode,vtkMRMLNode);
  void PrintSelf(ostream& os, vtkIndent indent);

  virtual vtkMRMLNode* CreateNodeInstance() = 0;

  // Description:
  // Set node attributes
  virtual void ReadXMLAttributes( const char** atts);

  // Description:
  // Write this node's information to a MRML file in XML format.
  virtual void WriteXML(ostream& of, int indent);

  // Description:
  // Copy the node's attributes to this object
  virtual void Copy(vtkMRMLNode *node);

  // Description:
  // Get node XML tag name (like Volume, Model)
  virtual const char* GetNodeTagName() = 0;

  // Description:
  // Finds the storage node and read the data
  virtual void UpdateScene(vtkMRMLScene *scene);

  // Description:
  // Numerical ID of the parent transform MRML node
  vtkSetStringMacro(ParentTransformNodeID);
  vtkGetStringMacro(ParentTransformNodeID);

  vtkGetObjectMacro(ParentTransformNode, vtkMRMLTransformNode);
  vtkSetObjectMacro(ParentTransformNode, vtkMRMLTransformNode);


  virtual int IsLinear() = 0;

  virtual vtkGeneralTransform* GetTransformToParent() {
     return this->TransformToParent; };

  int  IsTransformToWorldLinear() ;

  void GetTransformToWorld(vtkGeneralTransform* transformToWorld);

  int  IsTransformToNodeLinear(vtkMRMLTransformNode* node);

  void  GetTransformToNode(vtkMRMLTransformNode* node, vtkGeneralTransform* transformToNode);

  virtual void  GetTransformMatrixToWorld(vtkMatrix4x4* transformToWorld) = 0;

  virtual void  GetTransformMatrixToNode(vtkMRMLTransformNode* node, vtkMatrix4x4* transformToNode) = 0;

protected:
  vtkMRMLTransformNode();
  ~vtkMRMLTransformNode();
  vtkMRMLTransformNode(const vtkMRMLTransformNode&);
  void operator=(const vtkMRMLTransformNode&);

  char *ParentTransformNodeID;

  vtkMRMLTransformNode *ParentTransformNode;

  vtkGeneralTransform* TransformToParent;

};

#endif

