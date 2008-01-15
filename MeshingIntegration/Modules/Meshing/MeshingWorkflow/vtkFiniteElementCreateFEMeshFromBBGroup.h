/*=========================================================================

  Module:    $RCSfile: vtkFiniteElementCreateFEMeshFromBBGroup.h,v $

  Copyright (c) Kitware, Inc.
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkFiniteElementCreateFEMeshFromBBGroup - a tabbed notebook of UI pages
// .SECTION Description
// The class is derived from vtkKWMimxGroupBase. It contains 5 pages 1) Image
// 2) Surface 3) Bounding Box 4) F E Mesh 5) Mesh quality. Each page inturn
// contains a list of options specific to a page.

#ifndef __vtkFiniteElementCreateFEMeshFromBBGroup_h
#define __vtkFiniteElementCreateFEMeshFromBBGroup_h

#include "vtkBoundingBox.h"

#include "vtkKWMimxGroupBase.h"
#include "vtkKWMimxFEMeshMenuGroup.h"
#include "vtkKWMimxViewWindow.h"
#include "vtkKWMimxSurfaceMenuGroup.h"

class vtkKWComboBoxWithLabel;

class VTK_BOUNDINGBOX_EXPORT vtkFiniteElementCreateFEMeshFromBBGroup : public vtkKWMimxGroupBase
{
public:
  static vtkFiniteElementCreateFEMeshFromBBGroup* New();
  vtkTypeRevisionMacro(vtkFiniteElementCreateFEMeshFromBBGroup,vtkKWMimxGroupBase);
  void PrintSelf(ostream& os, vtkIndent indent);
  virtual void Update();
  virtual void UpdateEnableState();
  void CreateFEMeshFromBBCallback();
  void CreateFEMeshFromBBCancelCallback();
  void UpdateObjectLists();
protected:
        vtkFiniteElementCreateFEMeshFromBBGroup();
        ~vtkFiniteElementCreateFEMeshFromBBGroup();
        virtual void CreateWidget();
  vtkKWComboBoxWithLabel *SurfaceListComboBox;
  vtkKWComboBoxWithLabel *BBListComboBox;
private:
  vtkFiniteElementCreateFEMeshFromBBGroup(const vtkFiniteElementCreateFEMeshFromBBGroup&); // Not implemented
  void operator=(const vtkFiniteElementCreateFEMeshFromBBGroup&); // Not implemented
 };

#endif

