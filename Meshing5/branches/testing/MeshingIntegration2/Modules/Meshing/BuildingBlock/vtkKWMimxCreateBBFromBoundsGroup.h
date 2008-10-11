/*=========================================================================

Program:   MIMX Meshing Toolkit
Module:    $RCSfile: vtkKWMimxCreateBBFromBoundsGroup.h,v $
Language:  C++
Date:      $Date: 2008/02/16 00:15:33 $
Version:   $Revision: 1.9 $

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
// .NAME vtkKWMimxCreateBBFromBoundsGroup - a tabbed notebook of UI pages
// .SECTION Description
// The class is derived from vtkKWMimxGroupBase. It contains 5 pages 1) Image
// 2) Surface 3) Building Block 4) F E Mesh 5) Mesh quality. Each page inturn
// contains a list of options specific to a page.

#ifndef __vtkKWMimxCreateBBFromBoundsGroup_h
#define __vtkKWMimxCreateBBFromBoundsGroup_h

#include "vtkKWMimxGroupBase.h"
#include "vtkKWMimxFEMeshMenuGroup.h"
#include "vtkKWMimxMainWindow.h"
#include "vtkKWMimxSurfaceMenuGroup.h"

class vtkKWComboBoxWithLabel;

class vtkKWMimxCreateBBFromBoundsGroup : public vtkKWMimxGroupBase
{
public:
  static vtkKWMimxCreateBBFromBoundsGroup* New();
  vtkTypeRevisionMacro(vtkKWMimxCreateBBFromBoundsGroup,vtkKWMimxGroupBase);
  void PrintSelf(ostream& os, vtkIndent indent);
  virtual void Update();
  virtual void UpdateEnableState();
  void CreateBBFromBoundsDoneCallback();
  int CreateBBFromBoundsApplyCallback();
  void CreateBBFromBoundsCancelCallback();
  void UpdateObjectLists();
  void SelectionChangedCallback(const char*);
  vtkSetObjectMacro(SurfaceMenuGroup, vtkKWMimxSurfaceMenuGroup);
protected:
        vtkKWMimxCreateBBFromBoundsGroup();
        ~vtkKWMimxCreateBBFromBoundsGroup();
        virtual void CreateWidget();
  vtkKWComboBoxWithLabel *ObjectListComboBox;
  vtkKWMimxSurfaceMenuGroup *SurfaceMenuGroup;
private:
  vtkKWMimxCreateBBFromBoundsGroup(const vtkKWMimxCreateBBFromBoundsGroup&); // Not implemented
  void operator=(const vtkKWMimxCreateBBFromBoundsGroup&); // Not implemented
 };

#endif

