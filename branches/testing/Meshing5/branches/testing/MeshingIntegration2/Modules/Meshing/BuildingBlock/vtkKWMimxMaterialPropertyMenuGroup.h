/*=========================================================================

Program:   MIMX Meshing Toolkit
Module:    $RCSfile: vtkKWMimxMaterialPropertyMenuGroup.h,v $
Language:  C++
Date:      $Date: 2008/04/08 16:43:58 $
Version:   $Revision: 1.1 $

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
// .NAME vtkKWMimxMaterialPropertyMenuGroup - It is the base class for all Object menu options.
// .SECTION Description
// The class is derived from vtkKWMimxMainMenuGroup. It is the base class
// for all Object menu options.

#ifndef __vtkKWMimxMaterialPropertyMenuGroup_h
#define __vtkKWMimxMaterialPropertyMenuGroup_h

#include "vtkKWMimxMainMenuGroup.h"

class vtkKWMimxConstMatPropElSetGroup;
class vtkKWMimxApplyFEMeshMaterialPropertiesFromImageGroup;
class vtkKWMimxReBinFEMeshMaterialPropertiesFromImageGroup;

class vtkKWMimxMaterialPropertyMenuGroup : public vtkKWMimxMainMenuGroup
{
public:
  static vtkKWMimxMaterialPropertyMenuGroup* New();
  vtkTypeRevisionMacro(vtkKWMimxMaterialPropertyMenuGroup,vtkKWMimxMainMenuGroup);
  void PrintSelf(ostream& os, vtkIndent indent);
  virtual void Update();
  virtual void UpdateEnableState();  
  vtkSetObjectMacro(FEMeshList, vtkLinkedListWrapper);
  vtkSetObjectMacro(ImageList, vtkLinkedListWrapper);
  void UpdateObjectLists();
  void AssignMaterialPropertiesCallback();
  void AssignMaterialPropertiesElementSetCallback();
  void AssignImageBasedMaterialPropertiesCallback();
  void ReBinImageBasedMaterialPropertiesCallback();
protected:
        vtkKWMimxMaterialPropertyMenuGroup();
        ~vtkKWMimxMaterialPropertyMenuGroup();
        virtual void CreateWidget();
  vtkKWMimxConstMatPropElSetGroup *ConstMatPropElSetGroup;
  vtkKWMimxApplyFEMeshMaterialPropertiesFromImageGroup *ImageMatPropGroup;
  vtkKWMimxReBinFEMeshMaterialPropertiesFromImageGroup *ReBinImageMatPropGroup;
private:
  vtkKWMimxMaterialPropertyMenuGroup(const vtkKWMimxMaterialPropertyMenuGroup&); // Not implemented
  void operator=(const vtkKWMimxMaterialPropertyMenuGroup&); // Not implemented

};

#endif
