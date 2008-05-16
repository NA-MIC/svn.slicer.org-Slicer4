/*=========================================================================

Program:   MIMX Meshing Toolkit
Module:    $RCSfile: vtkKWMimxGroupBase.h,v $
Language:  C++
Date:      $Date: 2008/02/01 15:24:57 $
Version:   $Revision: 1.11 $

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
// .NAME vtkKWMimxGroupBase - a tabbed notebook of UI pages
// .SECTION Description
// The class is derived from vtkKWCompositeWidget. It contains 5 pages 1) Image
// 2) Surface 3) Building Block 4) F E Mesh 5) Mesh quality. Each page inturn
// contains a list of options specific to a page.

#ifndef __vtkKWMimxGroupBase_h
#define __vtkKWMimxGroupBase_h

#include "vtkKWCompositeWidget.h"
#include "vtkKWMimxFEMeshMenuGroup.h"
#include "vtkKWMimxMainWindow.h"
#include "vtkKWMimxSurfaceMenuGroup.h"

#include "vtkKWMimxViewProperties.h"

class vtkKWMenuButtonWithLabel;
class vtkKWFrameWithLabel;
class vtkKWPushButton;

class vtkLinkedListWrapper;
class vtkLinkedListWrapperTree;
class vtkKWComboBoxWithLabel;
class vtkKWMimxMainMenuGroup;
class vtkKWRenderWidget;

class vtkKWMimxGroupBase : public vtkKWCompositeWidget
{
public:
  static vtkKWMimxGroupBase* New();
  vtkTypeRevisionMacro(vtkKWMimxGroupBase,vtkKWCompositeWidget);
  void PrintSelf(ostream& os, vtkIndent indent);
  virtual void Update();
  virtual void UpdateEnableState();
  vtkSetObjectMacro(SurfaceList, vtkLinkedListWrapper);
  vtkSetObjectMacro(BBoxList, vtkLinkedListWrapper);
  vtkSetObjectMacro(FEMeshList, vtkLinkedListWrapper);
  vtkGetObjectMacro(MimxMainWindow, vtkKWMimxMainWindow);
  vtkSetObjectMacro(MimxMainWindow, vtkKWMimxMainWindow);
  vtkGetObjectMacro(MainFrame, vtkKWFrameWithLabel);
  vtkSetObjectMacro(ViewProperties, vtkKWMimxViewProperties);
  vtkSetObjectMacro(MenuGroup, vtkKWMimxMainMenuGroup);
  vtkSetObjectMacro(ImageList, vtkLinkedListWrapper);
  vtkSetObjectMacro(DoUndoTree, vtkLinkedListWrapperTree);
  vtkSetMacro(Count, int);
protected:
        vtkKWMimxGroupBase();
        virtual ~vtkKWMimxGroupBase();
        virtual void CreateWidget();
  vtkLinkedListWrapper *SurfaceList;
  vtkLinkedListWrapper *BBoxList;
  vtkLinkedListWrapper *FEMeshList;
  vtkLinkedListWrapper *ImageList;
  vtkKWPushButton *DoneButton;
  vtkKWPushButton *ApplyButton;
  vtkKWPushButton *CancelButton;
  vtkKWFrameWithLabel *MainFrame;

  vtkKWMimxMainWindow *MimxMainWindow;
  vtkKWMimxViewProperties *ViewProperties;
  vtkLinkedListWrapperTree *DoUndoTree;
  vtkKWMimxMainMenuGroup *MenuGroup;
  vtkIdType Count;  // to keep track of number of objects created during runtime

private:
  vtkKWMimxGroupBase(const vtkKWMimxGroupBase&); // Not implemented
  void operator=(const vtkKWMimxGroupBase&); // Not implemented
 };

#endif

