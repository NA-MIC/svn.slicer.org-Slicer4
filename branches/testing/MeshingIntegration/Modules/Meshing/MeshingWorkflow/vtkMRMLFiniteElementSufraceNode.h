/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkMRMLFiniteElementSurfaceNode.h,v $
  Date:      $Date: 2006/03/19 17:12:29 $
  Version:   $Revision: 1.3 $

=========================================================================auto=*/
#ifndef __vtkMRMLFiniteElementSurfaceNode_h
#define __vtkMRMLFiniteElementSurfaceNode_h

#include "vtkMRML.h"
#include "vtkMRMLNode.h"
#include "vtkMRMLStorageNode.h"

#include "vtkMatrix4x4.h"
#include "vtkTransform.h"
#include "vtkImageData.h"



class vtkImageData;

class VTK_SLICERDAEMON_EXPORT vtkMRMLFiniteElementSurfaceNode : public vtkMRMLModelNode
{
  public:
  static vtkMRMLFiniteElementSurfaceNode *New();
  vtkTypeMacro(vtkMRMLFiniteElementSurfaceNode,vtkMRMLModelNode);
  void PrintSelf(ostream& os, vtkIndent indent);

  virtual vtkMRMLNode* CreateNodeInstance();

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
  virtual const char* GetNodeTagName() {return "MeshingWorkflow";};

  vtkGetMacro(DataType, int);
  vtkGetObjectMacro(Actor, vtkActor);
 //BTX
   virtual double *GetBounds() {return NULL;}
   virtual void SetDataType(int ) = 0;
 //ETX
   vtkGetStringMacro(FilePath);
   vtkGetStringMacro(FileName);
   void SetFilePath(const char *InputFilePath);
   void SetFileName(const char *InputFileName);
   void SetObjectName(const char *FilterName, vtkIdType Count);

 
protected:
  vtkMRMLFiniteElementSurfaceNode();
  ~vtkMRMLFiniteElementSurfaceNode();
  vtkMRMLFiniteElementSurfaceNode(const vtkMRMLFiniteElementSurfaceNode&);
  void operator=(const vtkMRMLFiniteElementSurfaceNode&);



};

#endif

