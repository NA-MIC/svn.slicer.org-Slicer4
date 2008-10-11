/*=========================================================================

  Module:    $RCSfile: vtkFiniteElementMRMLMeshMenuGroup.h,v $

  Copyright (c) Kitware, Inc.
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkFiniteElementMRMLMeshMenuGroup - It is the base class for all Object menu options.
// .SECTION Description
// The class is derived from vtkKWMimxMainMenuGroup. It is the base class
// for all Object menu options.

#ifndef __vtkFiniteElementMRMLMeshMenuGroup_h
#define __vtkFiniteElementMRMLMeshMenuGroup_h

#include "vtkBoundingBox.h"

#include "vtkKWMimxMainMenuGroup.h"


class vtkKWMimxCreateBBFromBoundsGroup;
class vtkKWMimxCreateBBMeshSeedGroup;
class vtkFiniteElementCreateBBMeshSeedGroup;
class vtkKWMimxEditBBGroup;
class vtkFiniteElementCreateFEMeshFromBBGroup;
class vtkKWMimxViewProperties;
class vtkKWMimxSaveVTKBBGroup;
class vtkKWMimxEditBBMeshSeedGroup;
class vtkKWLoadSaveDialog;
class vtkKWMimxSaveVTKFEMeshGroup;
class vtkKWMimxDeleteObjectGroup;
class vtkKWMimxEditFEMeshLaplacianSmoothGroup;

// added for MRML storage of the saved information
class vtkMRMLScene; 

class VTK_BOUNDINGBOX_EXPORT vtkFiniteElementMRMLMeshMenuGroup : public vtkKWMimxMainMenuGroup
{
public:
  static vtkFiniteElementMRMLMeshMenuGroup* New();
  vtkTypeRevisionMacro(vtkFiniteElementMRMLMeshMenuGroup,vtkKWMimxMainMenuGroup);
  void PrintSelf(ostream& os, vtkIndent indent);
  
  virtual void Update();
  virtual void UpdateEnableState();
  virtual void BBMenuCallback();
  void FEMeshMenuCallback();
  void LoadFEMeshCallback();
  void LoadBBCallback();
  virtual void CreateBBCallback();
  void DeleteBBCallback();
  void CreateFEMeshCallback();
  void DeleteFEMeshCallback();
  void LoadVTKBBCallback();
  void LoadVTKFEMeshCallback();
  void CreateBBFromBoundsCallback();
  void BBMeshSeedMenuCallback();
  void CreateBBMeshSeedCallback();
  void EditBBMeshSeedCallback();
  void EditFEMeshCallback();
  void EditBBCallback();
  void CreateFEMeshFromBBCallback();
  void SaveBBCallback();
  void SaveFEMeshCallback();
  void SaveVTKBBCallback();
  void SaveVTKFEMeshCallback();
  
  vtkSetObjectMacro(SurfaceList, vtkLinkedListWrapper);
  vtkSetObjectMacro(FEMeshList, vtkLinkedListWrapper);
  void HideAllDialogBoxes();
  void SmoothLaplacianFEMeshCallback();
  
  // *** moved to public to allow invocation
  virtual void CreateWidget();
  
  // save reference to the scene to be used for storage 
   void SetMRMLSceneForStorage(vtkMRMLScene* scene);

protected:
    vtkFiniteElementMRMLMeshMenuGroup();
    ~vtkFiniteElementMRMLMeshMenuGroup();
  vtkKWMimxCreateBBFromBoundsGroup *CreateBBFromBounds;
  vtkFiniteElementCreateBBMeshSeedGroup *CreateBBMeshSeed;
  vtkKWMimxEditBBGroup *EditBB;
  vtkFiniteElementCreateFEMeshFromBBGroup *FEMeshFromBB;
  vtkKWMimxViewProperties *BBViewProperties;
  vtkKWMimxSaveVTKBBGroup *SaveVTKBBGroup;
  vtkKWMimxEditBBMeshSeedGroup *EditBBMeshSeedGroup;
  vtkKWMimxSaveVTKFEMeshGroup *SaveVTKFEMeshGroup;
  
  // added for MRML storage of the saved information
  vtkMRMLScene                 *savedMRMLScene;
  
vtkKWLoadSaveDialog *FileBrowserDialog;
vtkKWMimxDeleteObjectGroup *DeleteObjectGroup;
vtkKWMimxEditFEMeshLaplacianSmoothGroup *FEMeshLaplacianSmooth;
private:
  vtkFiniteElementMRMLMeshMenuGroup(const vtkFiniteElementMRMLMeshMenuGroup&); // Not implemented
  void operator=(const vtkFiniteElementMRMLMeshMenuGroup&); // Not implemented
 };

#endif
