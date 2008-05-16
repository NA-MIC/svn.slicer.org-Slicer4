/*=========================================================================

Program:   MIMX Meshing Toolkit
Module:    $RCSfile: vtkKWMimxBoundaryConditionsMenuGroup.h,v $
Language:  C++
Date:      $Date: 2008/04/22 22:55:57 $
Version:   $Revision: 1.3 $

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
// .NAME vtkKWMimxBoundaryConditionsMenuGroup - It is the base class for all Object menu options.
// .SECTION Description
// The class is derived from vtkKWMimxMainMenuGroup. It is the base class
// for all Object menu options.

#ifndef __vtkKWMimxBoundaryConditionsMenuGroup_h
#define __vtkKWMimxBoundaryConditionsMenuGroup_h

#include "vtkKWMimxMainMenuGroup.h"

class vtkKWMimxAssignBoundaryConditionsGroup;
class vtkKWMimxDeleteBoundaryConditionsGroup;

class vtkKWMimxBoundaryConditionsMenuGroup : public vtkKWMimxMainMenuGroup
{
public:
  static vtkKWMimxBoundaryConditionsMenuGroup* New();
  vtkTypeRevisionMacro(vtkKWMimxBoundaryConditionsMenuGroup,vtkKWMimxMainMenuGroup);
  void PrintSelf(ostream& os, vtkIndent indent);
  virtual void Update();
  virtual void UpdateEnableState();  
  vtkSetObjectMacro(FEMeshList, vtkLinkedListWrapper);
  void UpdateObjectLists();
  void AssignBoundaryConditionsCallback();
  void DeleteBoundaryConditionsCallback();
//  void AssignCallback();
protected:
        vtkKWMimxBoundaryConditionsMenuGroup();
        ~vtkKWMimxBoundaryConditionsMenuGroup();
        virtual void CreateWidget();
        vtkKWMimxAssignBoundaryConditionsGroup *AssignBoundaryConditionsGroup;
        vtkKWMimxDeleteBoundaryConditionsGroup *DeleteBoundaryConditionsGroup;
private:
  vtkKWMimxBoundaryConditionsMenuGroup(const vtkKWMimxBoundaryConditionsMenuGroup&); // Not implemented
  void operator=(const vtkKWMimxBoundaryConditionsMenuGroup&); // Not implemented

};

#endif
