/*=========================================================================

  Module:    $RCSfile: vtkFiniteElementMRMLCreateBBFromBoundsGroup.h,v $

  Copyright (c) Kitware, Inc.
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkFiniteElementMRMLCreateBBFromBoundsGroup - a tabbed notebook of UI pages
// .SECTION Description
// The class is derived from vtkKWMimxGroupBase. It contains 5 pages 1) Image
// 2) Surface 3) Bounding Box 4) F E Mesh 5) Mesh quality. Each page inturn
// contains a list of options specific to a page.

#ifndef __vtkFiniteElementMRMLCreateBBFromBoundsGroup_h
#define __vtkFiniteElementMRMLCreateBBFromBoundsGroup_h

#include "vtkBoundingBox.h"

#include "vtkKWMimxGroupBase.h"
#include "vtkKWMimxFEMeshMenuGroup.h"
#include "vtkKWMimxViewWindow.h"
#include "vtkKWMimxSurfaceMenuGroup.h"
#include "vtkKWMimxCreateBBFromBoundsGroup.h"

class vtkKWComboBoxWithLabel; 

class VTK_BOUNDINGBOX_EXPORT vtkFiniteElementMRMLCreateBBFromBoundsGroup : public vtkKWMimxCreateBBFromBoundsGroup
{
public:
  static vtkFiniteElementMRMLCreateBBFromBoundsGroup* New();
  vtkTypeRevisionMacro(vtkFiniteElementMRMLCreateBBFromBoundsGroup,vtkKWMimxCreateBBFromBoundsGroup);
  void PrintSelf(ostream& os, vtkIndent indent);
  virtual void Update();
  virtual void UpdateEnableState();
  virtual void CreateBBFromBoundsCallback();
  virtual void CreateBBFromBoundsCancelCallback();
  virtual void UpdateObjectLists();
protected:
        vtkFiniteElementMRMLCreateBBFromBoundsGroup();
        ~vtkFiniteElementMRMLCreateBBFromBoundsGroup();
        virtual void CreateWidget();
  vtkKWComboBoxWithLabel *ObjectListComboBox;
private:
  vtkFiniteElementMRMLCreateBBFromBoundsGroup(const vtkFiniteElementMRMLCreateBBFromBoundsGroup&); // Not implemented
  void operator=(const vtkFiniteElementMRMLCreateBBFromBoundsGroup&); // Not implemented
 };

#endif

