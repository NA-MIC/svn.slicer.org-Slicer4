/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkMRMLMeshingWorkflowNode.h,v $
  Date:      $Date: 2006/03/19 17:12:29 $
  Version:   $Revision: 1.3 $

=========================================================================auto=*/
#ifndef __vtkMRMLMeshingWorkflowNode_h
#define __vtkMRMLMeshingWorkflowNode_h

#include "vtkMRML.h"
#include "vtkMRMLNode.h"
#include "vtkMRMLStorageNode.h" 

#include "vtkMatrix4x4.h"
#include "vtkTransform.h"
#include "vtkImageData.h"

#include "vtkMeshingWorkflow.h"

class vtkImageData;

class VTK_SLICERDAEMON_EXPORT vtkMRMLMeshingWorkflowNode : public vtkMRMLNode
{
  public:
  static vtkMRMLMeshingWorkflowNode *New();
  vtkTypeMacro(vtkMRMLMeshingWorkflowNode,vtkMRMLNode);
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


  vtkGetMacro(NumberOfIterations, int);
  vtkSetMacro(NumberOfIterations, int);

  vtkGetMacro(Conductance, double);
  vtkSetMacro(Conductance, double);


  vtkGetMacro(TimeStep, double);
  vtkSetMacro(TimeStep, double);
 
  vtkGetStringMacro(InputVolumeRef);
  vtkSetStringMacro(InputVolumeRef);
  
  vtkGetStringMacro(OutputVolumeRef);
  vtkSetStringMacro(OutputVolumeRef);

 
protected:
  vtkMRMLMeshingWorkflowNode();
  ~vtkMRMLMeshingWorkflowNode();
  vtkMRMLMeshingWorkflowNode(const vtkMRMLMeshingWorkflowNode&);
  void operator=(const vtkMRMLMeshingWorkflowNode&);

  double Conductance;
  double TimeStep;
  int NumberOfIterations;
  
  char* InputVolumeRef;
  char* OutputVolumeRef;

};

#endif

