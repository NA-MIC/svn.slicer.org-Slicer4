/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkMRMLFiniteElementMeshNode.h,v $
  Date:      $Date: 2006/03/19 17:12:29 $
  Version:   $Revision: 1.3 $

=========================================================================auto=*/
#ifndef __vtkMRMLFiniteElementMeshNode_h
#define __vtkMRMLFiniteElementMeshNode_h

//#include "vtkMRML.h"
#include "vtkMRMLUnstructuredGridNode.h"



class VTK_MRML_EXPORT vtkMRMLFiniteElementMeshNode : public vtkMRMLUnstructuredGridNode
{
  public:
  static vtkMRMLFiniteElementMeshNode *New();
  vtkTypeMacro(vtkMRMLFiniteElementMeshNode,vtkMRMLUnstructuredGridNode);
  void PrintSelf(ostream& os, vtkIndent indent);

  virtual vtkMRMLFiniteElementMeshNode* CreateNodeInstance();

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
  virtual const char* GetNodeTagName() {return "FiniteElementMesh";};

  vtkGetMacro(DataType, int);
  vtkSetMacro(DataType, int);

  vtkGetStringMacro(FileName);
  vtkSetStringMacro(FileName);
  
  vtkGetStringMacro(FilePath);
  vtkSetStringMacro(FilePath);
  
protected:
  vtkMRMLFiniteElementMeshNode();
  ~vtkMRMLFiniteElementMeshNode();
  vtkMRMLFiniteElementMeshNode(const vtkMRMLFiniteElementMeshNode&);
  void operator=(const vtkMRMLFiniteElementMeshNode&);

  int   DataType;  
  char* FileName;
  char* FilePath;
};

#endif

