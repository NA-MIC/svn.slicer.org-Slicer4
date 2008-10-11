/*=========================================================================

Program:   MIMX Meshing Toolkit
Module:    $RCSfile: vtkKWMimxCreateFEMeshFromBBGroup.h,v $
Language:  C++
Date:      $Date: 2008/04/22 16:56:18 $
Version:   $Revision: 1.10 $

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
// .NAME vtkKWMimxCreateFEMeshFromBBGroup - a tabbed notebook of UI pages
// .SECTION Description
// The class is derived from vtkKWMimxGroupBase. It contains 5 pages 1) Image
// 2) Surface 3) Building Block 4) F E Mesh 5) Mesh quality. Each page inturn
// contains a list of options specific to a page.

#ifndef __vtkKWMimxCreateFEMeshFromBBGroup_h
#define __vtkKWMimxCreateFEMeshFromBBGroup_h

#include "vtkKWMimxGroupBase.h"
#include "vtkKWMimxFEMeshMenuGroup.h"
#include "vtkKWMimxMainWindow.h"
#include "vtkKWMimxSurfaceMenuGroup.h"

class vtkKWComboBoxWithLabel;
class vtkIntArray;
class vtkKWEntryWithLabel;
class vtkKWMimxNodeElementNumbersGroup;


class vtkKWMimxCreateFEMeshFromBBGroup : public vtkKWMimxGroupBase
{
public:
  static vtkKWMimxCreateFEMeshFromBBGroup* New();
  vtkTypeRevisionMacro(vtkKWMimxCreateFEMeshFromBBGroup,vtkKWMimxGroupBase);
  void PrintSelf(ostream& os, vtkIndent indent);
  virtual void Update();
  virtual void UpdateEnableState();
  void CreateFEMeshFromBBDoneCallback();
  int CreateFEMeshFromBBApplyCallback();
  void CreateFEMeshFromBBCancelCallback();
  void UpdateObjectLists();
protected:
        vtkKWMimxCreateFEMeshFromBBGroup();
        ~vtkKWMimxCreateFEMeshFromBBGroup();
        virtual void CreateWidget();
  vtkKWComboBoxWithLabel *SurfaceListComboBox;
  vtkKWComboBoxWithLabel *BBListComboBox;
  vtkIntArray *OriginalPosition;
  vtkKWEntryWithLabel *NodeNumberEntry;
  vtkKWEntryWithLabel *ElementSetNameEntry;
  vtkKWEntryWithLabel *ElementNumberEntry;
  vtkKWMimxNodeElementNumbersGroup *NodeElementNumbersGroup;
private:
  vtkKWMimxCreateFEMeshFromBBGroup(const vtkKWMimxCreateFEMeshFromBBGroup&); // Not implemented
  void operator=(const vtkKWMimxCreateFEMeshFromBBGroup&); // Not implemented
 };

#endif

