/*=========================================================================

Program:   MIMX Meshing Toolkit
Module:    $RCSfile: vtkMimxSurfacePolyDataActor.h,v $
Language:  C++
Date:      $Date: 2008/01/14 21:07:52 $
Version:   $Revision: 1.8 $

 Musculoskeletal Imaging, Modelling and Experimentation (MIMX)
 Center for Computer Aided Design
 The University of Iowa
 Iowa City, IA 52242
 http://www.ccad.uiowa.edu/mimx/
 
Copyright (c) The University of Iowa. All rights reserved.
See MIMXCopyright.txt or http://www.ccad.uiowa.edu/mimx/Copyright.htm for details.

This software is distributed WITHOUT ANY WARRANTY; without even 
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
PURPOSE.  See the above copyright notices for more information.

=========================================================================*/

// .NAME vtkMimxSurfacePolyDataActor - a 3D non-orthogonal axes representation
// .SECTION Description
// vtkMimxSurfacePolyDataActor is the abstract base class for all the pipeline setup for
// different types of datatypes. Data types concidered are vtkPolyData,
// vtkStructuredGrid (both plane and solid) and vtkUnstructuredGrid.

#ifndef __vtkMimxSurfacePolyDataActor_h
#define __vtkMimxSurfacePolyDataActor_h

#include "vtkMimxActorBase.h"

class vtkActor;
class vtkPolyData;
class vtkPolyDataMapper;

class vtkMimxSurfacePolyDataActor : public vtkMimxActorBase
{
public:
  static vtkMimxSurfacePolyDataActor *New();
  vtkTypeRevisionMacro(vtkMimxSurfacePolyDataActor,vtkMimxActorBase);
  void PrintSelf(ostream& os, vtkIndent indent);
 vtkPolyData* GetDataSet();
// void SetDataType(int){};
 //vtkSetMacro(PolyData, vtkPolyData*);
 //vtkGetMacro(PolyData, vtkPolyData*);
  // Description:   
protected:
  vtkMimxSurfacePolyDataActor();
  ~vtkMimxSurfacePolyDataActor();
  vtkPolyData *PolyData;
  vtkPolyDataMapper *PolyDataMapper;
private:
  vtkMimxSurfacePolyDataActor(const vtkMimxSurfacePolyDataActor&);  // Not implemented.
  void operator=(const vtkMimxSurfacePolyDataActor&);  // Not implemented.
};

#endif

