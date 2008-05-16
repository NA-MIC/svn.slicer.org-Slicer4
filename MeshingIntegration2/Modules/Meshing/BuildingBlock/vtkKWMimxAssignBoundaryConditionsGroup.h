/*=========================================================================

Program:   MIMX Meshing Toolkit
Module:    $RCSfile: vtkKWMimxAssignBoundaryConditionsGroup.h,v $
Language:  C++
Date:      $Date: 2008/05/05 19:30:08 $
Version:   $Revision: 1.4 $

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
// .NAME vtkKWMimxAssignBoundaryConditionsGroup - a tabbed notebook of UI pages
// .SECTION Description
// The class is derived from vtkKWMimxGroupBase. It contains 5 pages 1) Image
// 2) Surface 3) Building Block 4) F E Mesh 5) Mesh quality. Each page inturn
// contains a list of options specific to a page.

#ifndef __vtkKWMimxAssignBoundaryConditionsGroup_h
#define __vtkKWMimxAssignBoundaryConditionsGroup_h

#include "vtkKWMimxGroupBase.h"
#include "vtkKWMimxMainWindow.h"

class vtkKWComboBoxWithLabel;
class vtkKWEntryWithLabel;
class vtkKWCheckButtonWithLabel;
class vtkKWFrameWithLabel;

class vtkKWMimxAssignBoundaryConditionsGroup : public vtkKWMimxGroupBase
{
public:
  static vtkKWMimxAssignBoundaryConditionsGroup* New();
vtkTypeRevisionMacro(vtkKWMimxAssignBoundaryConditionsGroup,vtkKWMimxGroupBase);
void PrintSelf(ostream& os, vtkIndent indent);
virtual void Update();
virtual void UpdateEnableState();
//void AssignBoundaryConditionsDoneCallback();
void AssignBoundaryConditionsCancelCallback();
int AssignBoundaryConditionsApplyCallback();
void UpdateObjectLists();
void SelectionChangedCallback(const char *Selection);
void NodeSetChangedCallback(const char *Selection);
void BoundaryConditionTypeSelectionChangedCallback(const char *Selection);
void StepNumberChangedCallback(const char *StepNum);
void AddStepNumberCallback();
void ViewBoundaryConditionsCallback(int Mode);
protected:
        vtkKWMimxAssignBoundaryConditionsGroup();
        ~vtkKWMimxAssignBoundaryConditionsGroup();
        void GetValue();
        virtual void CreateWidget();
vtkKWComboBoxWithLabel *ObjectListComboBox;
vtkKWComboBoxWithLabel *NodeSetComboBox;
vtkKWComboBoxWithLabel *BoundaryConditionTypeComboBox;
vtkKWFrameWithLabel *StepFrame;
vtkKWFrameWithLabel *DirectionFrame;
vtkKWEntryWithLabel *DirectionXEntry;
vtkKWEntryWithLabel *DirectionYEntry;
vtkKWEntryWithLabel *DirectionZEntry;
vtkKWComboBoxWithLabel *StepNumberComboBox;
vtkKWPushButton *AddStepPushButton;
vtkKWCheckButtonWithLabel *ViewBoundaryConditionsButton;
void ConcatenateStrings(const char*, const char*, 
                                                const char*, const char*, const char*, char*);
int IsStepEmpty(vtkUnstructuredGrid *ugrid);
int CancelStatus;
vtkActor *GlyphActor;
vtkKWFrameWithLabel *ViewFrame;
vtkKWComboBoxWithLabel *ViewDirectionComboBox;
private:
  vtkKWMimxAssignBoundaryConditionsGroup(const vtkKWMimxAssignBoundaryConditionsGroup&); // Not implemented
void operator=(const vtkKWMimxAssignBoundaryConditionsGroup&); // Not implemented
 };

#endif

