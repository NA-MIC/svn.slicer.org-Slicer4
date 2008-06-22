/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkMRMLFiniteElementBuildingBlockNode.h,v $
  Date:      $Date: 2006/03/19 17:12:29 $
  Version:   $Revision: 1.3 $

=========================================================================auto=*/
#ifndef __vtkMRMLFiniteElementBuildingBlockNode_h
#define __vtkMRMLFiniteElementBuildingBlockNode_h

//#include "vtkMRML.h"
#include "vtkMRMLUnstructuredGridNode.h"



class VTK_MRML_EXPORT vtkMRMLFiniteElementBuildingBlockNode : public vtkMRMLUnstructuredGridNode
{
  public:
  static vtkMRMLFiniteElementBuildingBlockNode *New();
  vtkTypeMacro(vtkMRMLFiniteElementBuildingBlockNode,vtkMRMLUnstructuredGridNode);
  void PrintSelf(ostream& os, vtkIndent indent);

  virtual vtkMRMLFiniteElementBuildingBlockNode* CreateNodeInstance();

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
  virtual const char* GetNodeTagName() {return "FiniteElementBuildingBlock";};

  vtkGetMacro(DataType, int);
  vtkSetMacro(DataType, int);

  vtkGetStringMacro(FileName);
 // vtkSetStringMacro(FileName);
  
  vtkGetStringMacro(FilePath);
//  vtkSetStringMacro(FilePath);
  
  void  SetFileName(char* aString) { strcpy(FileName,aString); }
  void  SetFilePath(char* aString) { strcpy(FilePath,aString); }
 
  
protected:
  vtkMRMLFiniteElementBuildingBlockNode();
  ~vtkMRMLFiniteElementBuildingBlockNode();
  vtkMRMLFiniteElementBuildingBlockNode(const vtkMRMLFiniteElementBuildingBlockNode&);
  void operator=(const vtkMRMLFiniteElementBuildingBlockNode&);

  int   DataType;  
  char* FileName;
  char* FilePath;
};

#endif

