/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkMeshingWorkflowLogic.h,v $
  Date:      $Date: 2006/03/19 17:12:29 $
  Version:   $Revision: 1.3 $

=========================================================================auto=*/
#ifndef __vtkMeshingWorkflowLogic_h
#define __vtkMeshingWorkflowLogic_h

#include "vtkSlicerModuleLogic.h"
#include "vtkMRMLScene.h"

#include "vtkMeshingWorkflow.h"
#include "vtkMRMLMeshingWorkflowNode.h"


class VTK_SLICERDAEMON_EXPORT vtkMeshingWorkflowLogic : public vtkSlicerModuleLogic
{
  public:
  static vtkMeshingWorkflowLogic *New();
  vtkTypeMacro(vtkMeshingWorkflowLogic,vtkSlicerModuleLogic);
  void PrintSelf(ostream& os, vtkIndent indent);

  // TODO: do we need to observe MRML here?
  virtual void ProcessMrmlEvents ( vtkObject *caller, unsigned long event,
                                   void *callData ){};

  // Description: Get/Set MRML node
  vtkGetObjectMacro (MeshingWorkflowNode, vtkMRMLMeshingWorkflowNode);
  vtkSetObjectMacro (MeshingWorkflowNode, vtkMRMLMeshingWorkflowNode);
  
  void Apply();
  
protected:
  vtkMeshingWorkflowLogic();
  ~vtkMeshingWorkflowLogic();
  vtkMeshingWorkflowLogic(const vtkMeshingWorkflowLogic&);
  void operator=(const vtkMeshingWorkflowLogic&);

  vtkMRMLMeshingWorkflowNode* MeshingWorkflowNode;
  //vtkITKMeshingWorkflow* MeshingWorkflow;


};

#endif

