/*=========================================================================

  Module:    $RCSfile: vtkMeshingWorkflowMRMLNotebook.h,v $

  Copyright (c) Kitware, Inc.
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkMeshingWorkflowMRMLNotebook - a tabbed notebook of UI pages
// .SECTION Description
// The class is derived from vtkKWCompositeWidget. It contains 5 pages 1) Image
// 2) Surface 3) Bounding Box 4) F E Mesh 5) Mesh quality. Each page inturn
// contains a list of options specific to a page.

#ifndef __vtkMeshingWorkflowMRMLNotebook_h
#define __vtkMeshingWorkflowMRMLNotebook_h

#include "vtkBoundingBox.h"

#include "vtkKWCompositeWidget.h"
#include "vtkKWMimxSurfaceMenuGroup.h"
#include "vtkKWMimxViewWindow.h"
#include "vtkKWMimxFEMeshMenuGroup.h"


class vtkKWNotebook;
class vtkKWFrameWithScrollbar;
class vtkKWMenuButtonWithLabel;
class vtkMRMLScene;
class vtkFiniteElementMRMLMeshMenuGroup;
class vtkKWMimxMainWindow;
class vtkKWMimxBBMenuGroup;
class vtkKWMimxImageMenuGroup;
class vtkLinkedListWrapperTree;
class vtkKWMimxQualityMenuGroup;
class vtkKWMimxMaterialPropertyMenuGroup;

class  vtkMeshingWorkflowMRMLNotebook : public vtkKWCompositeWidget
{
public:
  static vtkMeshingWorkflowMRMLNotebook* New();
  vtkTypeRevisionMacro(vtkMeshingWorkflowMRMLNotebook,vtkKWCompositeWidget);
  void PrintSelf(ostream& os, vtkIndent indent);
 // const char* GetActivePage();        // return the name of the chosen page
 // int GetActiveOption();        // return the chosen operation
  virtual void Update();
  virtual void UpdateEnableState();
  vtkSetObjectMacro(MimxViewWindow, vtkKWMimxViewWindow);
  vtkGetObjectMacro(MimxViewWindow, vtkKWMimxViewWindow);

 
  
  // these VTK macro-based definitions caused a seg fault when the value was set, so implemented the 
  // methods directly instead of using the VTK macros

  //vtkSetObjectMacro(DoUndoTree, vtkLinkedListWrapperTree);
  //vtkGetObjectMacro(DoUndoTree, vtkLinkedListWrapperTree);
  //vtkSetObjectMacro(MimxMainWindow, vtkKWMimxMainWindow);
  //vtkGetObjectMacro(MimxMainWindow, vtkKWMimxMainWindow);
  virtual void SetDoUndoTree(vtkLinkedListWrapperTree*);
  virtual vtkLinkedListWrapperTree* GetDoUndoTree();
  virtual void SetMimxMainWindow(vtkKWMimxMainWindow*);
  virtual vtkKWMimxMainWindow* GetMimxMainWindow();
  
  // 
  // save reference to the scene to be used for storage 
   void SetMRMLSceneForStorage(vtkMRMLScene* scene);
   
protected:
        vtkMeshingWorkflowMRMLNotebook();
        ~vtkMeshingWorkflowMRMLNotebook();
  void SetLists();
        virtual void CreateWidget();
        vtkKWNotebook *Notebook;
        vtkKWMimxViewWindow *MimxViewWindow;
        vtkKWMimxMainWindow *MimxMainWindow;
        vtkKWMimxSurfaceMenuGroup *SurfaceMenuGroup;
        vtkKWMimxFEMeshMenuGroup *FEMeshMenuGroup;
        vtkKWMimxBBMenuGroup *BBMenuGroup;
        vtkKWMimxImageMenuGroup *ImageMenuGroup;
        vtkKWMimxQualityMenuGroup *QualityMenuGroup;
        vtkKWMimxMaterialPropertyMenuGroup *MaterialPropertyMenuGroup;
        vtkLinkedListWrapperTree *DoUndoTree;
        vtkMRMLScene             *savedMRMLScene;
private:
  vtkMeshingWorkflowMRMLNotebook(const vtkMeshingWorkflowMRMLNotebook&); // Not implemented
  void operator=(const vtkMeshingWorkflowMRMLNotebook&); // Not implemented
 };

#endif

