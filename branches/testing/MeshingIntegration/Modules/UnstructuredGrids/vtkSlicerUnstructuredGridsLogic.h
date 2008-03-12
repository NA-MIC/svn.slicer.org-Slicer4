/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkSlicerUnstructuredGridsLogic.h,v $
  Date:      $Date: 2006/01/08 04:48:05 $
  Version:   $Revision: 1.45 $

============================f=============================================auto=*/

// .NAME vtkSlicerUnstructuredGridsLogic - slicer logic class for volumes manipulation
// .SECTION Description
// This class manages the logic associated with reading, saving,
// and changing propertied of the volumes


#ifndef __vtkSlicerUnstructuredGridsLogic_h
#define __vtkSlicerUnstructuredGridsLogic_h

#include <stdlib.h>

#include "vtkSlicerBaseLogic.h"
#include "vtkSlicerLogic.h"

#include "vtkMRML.h"
#include "vtkMRMLUnstructuredGridNode.h"


class VTK_SLICER_BASE_LOGIC_EXPORT vtkSlicerUnstructuredGridsLogic : public vtkSlicerLogic 
{
  public:
  
  // The Usual vtk class functions
  static vtkSlicerUnstructuredGridsLogic *New();
  vtkTypeRevisionMacro(vtkSlicerUnstructuredGridsLogic,vtkObject);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // The currently active mrml volume node 
  vtkGetObjectMacro (ActiveUnstructuredGridNode, vtkMRMLUnstructuredGridNode);
  void SetActiveUnstructuredGridNode (vtkMRMLUnstructuredGridNode *ActiveUnstructuredGridNode);

  // Description:
  // Create new mrml UnstructuredGrid node and
  // read it's polydata from a specified file
  vtkMRMLUnstructuredGridNode* AddUnstructuredGrid (const char* filename);

  // Description:
  // Create UnstructuredGrid nodes and
  // read their polydata from a specified directory
  int AddUnstructuredGrids (const char* dirname, const char* suffix );

  // Description:
  // Write UnstructuredGrid's polydata  to a specified file
  int SaveUnstructuredGrid (const char* filename, vtkMRMLUnstructuredGridNode *UnstructuredGridNode);

  // Description:
  // Read in a scalar overlay and add it to the UnstructuredGrid node
  int AddScalar(const char* filename, vtkMRMLUnstructuredGridNode *UnstructuredGridNode);

  // Description:
  // Update logic state when MRML scene chenges
  virtual void ProcessMRMLEvents ( vtkObject * /*caller*/, 
                                  unsigned long /*event*/, 
                                  void * /*callData*/ );    
protected:
  vtkSlicerUnstructuredGridsLogic();
  ~vtkSlicerUnstructuredGridsLogic();
  vtkSlicerUnstructuredGridsLogic(const vtkSlicerUnstructuredGridsLogic&);
  void operator=(const vtkSlicerUnstructuredGridsLogic&);

  // Description:
  //
  vtkMRMLUnstructuredGridNode *ActiveUnstructuredGridNode;
};

#endif

