/*=========================================================================

Program:   MIMX Meshing Toolkit
Module:    $RCSfile: vtkKWMimxNodeElementNumbersGroup.h,v $
Language:  C++
Date:      $Date: 2008/04/22 16:56:18 $
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
// .NAME vtkKWMimxNodeElementNumbersGroup - a taFEMeshed notebook of UI pages
// .SECTION Description
// The class is derived from vtkKWMimxGroupBase. It contains 5 pages 1) Image
// 2) Surface 3) Building Block 4) F E Mesh 5) Mesh quality. Each page inturn
// contains a list of options specific to a page.

#ifndef __vtkKWMimxNodeElementNumbersGroup_h
#define __vtkKWMimxNodeElementNumbersGroup_h

#include "vtkKWMimxGroupBase.h"
#include "vtkKWMimxFEMeshMenuGroup.h"
#include "vtkKWMimxMainWindow.h"
#include "vtkKWMimxSurfaceMenuGroup.h"

class vtkKWComboBoxWithLabel;
class vtkKWEntryWithLabel;
class vtkKWCheckButtonWithLabel;

class vtkKWMimxNodeElementNumbersGroup : public vtkKWMimxGroupBase
{
public:
  static vtkKWMimxNodeElementNumbersGroup* New();
vtkTypeRevisionMacro(vtkKWMimxNodeElementNumbersGroup,vtkKWMimxGroupBase);
void PrintSelf(ostream& os, vtkIndent indent);
virtual void Update();
virtual void UpdateEnableState();
//void EditNodeElementNumbersDoneCallback();
//void EditNodeElementNumbersCancelCallback();
//int EditNodeElementNumbersApplyCallback();
//void UpdateObjectLists();
//void SelectionChangedCallback(const char *);
//void ApplyNodeNumbersCallback(int);
//void ApplyElementNumbersCallback(int);
vtkGetObjectMacro(NodeNumberEntry, vtkKWEntryWithLabel);
vtkGetObjectMacro(ElementNumberEntry, vtkKWEntryWithLabel);
vtkGetObjectMacro(NodeSetNameEntry, vtkKWEntryWithLabel);
vtkGetObjectMacro(ElementSetNameEntry, vtkKWEntryWithLabel);

protected:
        vtkKWMimxNodeElementNumbersGroup();
        ~vtkKWMimxNodeElementNumbersGroup();
        virtual void CreateWidget();
        vtkKWComboBoxWithLabel *ObjectListComboBox;
        vtkKWEntryWithLabel *NodeNumberEntry;
        vtkKWEntryWithLabel *ElementNumberEntry;
        vtkKWEntryWithLabel *ElementSetNameEntry;
        vtkKWEntryWithLabel *NodeSetNameEntry;
        vtkKWCheckButtonWithLabel *NodeNumberCheckButton;
        vtkKWCheckButtonWithLabel *ElementNumberCheckButton;
private:
  vtkKWMimxNodeElementNumbersGroup(const vtkKWMimxNodeElementNumbersGroup&); // Not implemented
void operator=(const vtkKWMimxNodeElementNumbersGroup&); // Not implemented
 };

#endif

