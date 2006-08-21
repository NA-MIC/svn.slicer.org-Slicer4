/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkSlicerModelsLogic.h,v $
  Date:      $Date: 2006/01/08 04:48:05 $
  Version:   $Revision: 1.45 $

=========================================================================auto=*/

// .NAME vtkSlicerModelsLogic - slicer logic class for volumes manipulation
// .SECTION Description
// This class manages the logic associated with reading, saving,
// and changing propertied of the volumes


#ifndef __vtkSlicerModelsLogic_h
#define __vtkSlicerModelsLogic_h

#include <stdlib.h>

#include "vtkSlicerBaseLogic.h"
#include "vtkSlicerLogic.h"

#include "vtkMRML.h"
#include "vtkMRMLModelNode.h"


class VTK_SLICER_BASE_LOGIC_EXPORT vtkSlicerModelsLogic : public vtkSlicerLogic 
{
  public:
  
  // The Usual vtk class functions
  static vtkSlicerModelsLogic *New();
  vtkTypeRevisionMacro(vtkSlicerModelsLogic,vtkObject);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // The currently active mrml volume node 
  vtkGetObjectMacro (ActiveModelNode, vtkMRMLModelNode);
  void SetActiveModelNode (vtkMRMLModelNode *ActiveModelNode);

  // Description:
  // Create new mrml node and associated storage node.
  // Read image data from a specified file
  vtkMRMLModelNode* AddModel (char* filename);

  // Description:
  // Write model's polydata  to a specified file
  int SaveModel (char* filename, vtkMRMLModelNode *modelNode);

  // Description:
  // Update logic state when MRML scene chenges
  virtual void ProcessMRMLEvents ( vtkObject * /*caller*/, 
                                  unsigned long /*event*/, 
                                  void * /*callData*/ );    
protected:
  vtkSlicerModelsLogic();
  ~vtkSlicerModelsLogic();
  vtkSlicerModelsLogic(const vtkSlicerModelsLogic&);
  void operator=(const vtkSlicerModelsLogic&);

  // Description:
  //
  vtkMRMLModelNode *ActiveModelNode;
};

#endif

