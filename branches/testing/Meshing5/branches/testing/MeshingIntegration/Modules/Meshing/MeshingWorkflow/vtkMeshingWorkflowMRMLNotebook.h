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
#include "vtkKWMimxFEMeshMenuGroup.h"
#include "vtkKWMimxViewWindow.h"
#include "vtkFESurfaceMRMLMenuGroup.h"


class vtkKWNotebook;
class vtkKWFrameWithScrollbar;
class vtkKWMenuButtonWithLabel;
class vtkMRMLScene;
class vtkFiniteElementMRMLMeshMenuGroup;

class VTK_BOUNDINGBOX_EXPORT vtkMeshingWorkflowMRMLNotebook : public vtkKWCompositeWidget
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

  // save reference to the scene to be used for storage 
   void SetMRMLSceneForStorage(vtkMRMLScene* scene);
   
protected:
        vtkMeshingWorkflowMRMLNotebook();
        ~vtkMeshingWorkflowMRMLNotebook();
  void SetLists();
        virtual void CreateWidget();
        vtkKWNotebook *Notebook;
        vtkKWMimxViewWindow *MimxViewWindow;
        vtkFESurfaceMRMLMenuGroup *SurfaceMenuGroup;
        vtkFiniteElementMRMLMeshMenuGroup *FEMeshMenuGroup;
        vtkMRMLScene             *savedMRMLScene;
private:
  vtkMeshingWorkflowMRMLNotebook(const vtkMeshingWorkflowMRMLNotebook&); // Not implemented
  void operator=(const vtkMeshingWorkflowMRMLNotebook&); // Not implemented
 };

#endif

