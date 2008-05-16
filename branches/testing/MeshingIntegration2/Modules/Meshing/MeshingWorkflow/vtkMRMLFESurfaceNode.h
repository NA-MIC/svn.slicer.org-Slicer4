/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkMRMLFESurfaceNode.h,v $
  Date:      $Date: 2006/03/19 17:12:29 $
  Version:   $Revision: 1.3 $

=========================================================================auto=*/
#ifndef __vtkMRMLFESurfaceNode_h
#define __vtkMRMLFESurfaceNode_h

#include "vtkMRML.h"
#include "vtkMRMLModelNode.h"
//#include "vtkMRMLStorageNode.h"


class VTK_MRML_EXPORT vtkMRMLFESurfaceNode : public vtkMRMLModelNode
{
  public:
  static vtkMRMLFESurfaceNode *New();
  vtkTypeMacro(vtkMRMLFESurfaceNode,vtkMRMLModelNode);
  void PrintSelf(ostream& os, vtkIndent indent);

  virtual vtkMRMLModelNode* CreateNodeInstance();

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
  virtual const char* GetNodeTagName() {return "FiniteElementSurface";};

  vtkGetMacro(DataType, int);
  vtkSetMacro(DataType, int);

  vtkGetStringMacro(FileName);
  vtkSetStringMacro(FileName);
  
  vtkGetStringMacro(FilePath);
  vtkSetStringMacro(FilePath);
  
  //void  SetSurfaceFileName(char* aString) { strcpy(SurfaceFileName,aString); }
  //void  SetSurfaceFilePath(char* aString) { strcpy(SurfaceFilePath,aString); }
  
protected:
  vtkMRMLFESurfaceNode();
  ~vtkMRMLFESurfaceNode();
  vtkMRMLFESurfaceNode(const vtkMRMLFESurfaceNode&);
  void operator=(const vtkMRMLFESurfaceNode&);

  int   DataType;  
  char *FileName;
  char *FilePath;
};

#endif

