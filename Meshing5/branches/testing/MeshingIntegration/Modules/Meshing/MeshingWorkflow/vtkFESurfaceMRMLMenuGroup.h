/*=========================================================================

  Module:    $RCSfile: vtkFESurfaceMRMLMenuGroup.h,v $

  Copyright (c) Kitware, Inc.
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkFESurfaceMRMLMenuGroup - It is the base class for all Object menu options.
// .SECTION Description
// The class is derived from vtkFESurfaceMRMLMenuGroup. This has identical, but uses
// MRML-based storage instead of local list storage. 

#ifndef __vtkFESurfaceMRMLMenuGroup_h
#define __vtkFESurfaceMRMLMenuGroup_h

#include "vtkBoundingBox.h"

#include "vtkKWMimxSurfaceMenuGroup.h"

class vtkKWFileBrowserDialog;
class vtkKWMimxSaveSTLSurfaceGroup;
class vtkKWMimxSaveVTKSurfaceGroup;
class vtkKWMimxDeleteObjectGroup;
class vtkMRMLScene; 

class VTK_BOUNDINGBOX_EXPORT vtkFESurfaceMRMLMenuGroup : public vtkKWMimxSurfaceMenuGroup
{
public:
  static vtkFESurfaceMRMLMenuGroup* New();
  vtkTypeRevisionMacro(vtkFESurfaceMRMLMenuGroup,vtkKWMimxMainMenuGroup);
  void PrintSelf(ostream& os, vtkIndent indent);
  virtual void Update();
  virtual void UpdateEnableState();
  virtual void SurfaceMenuCallback();
  virtual void LoadSurfaceCallback();
  virtual void LoadSTLSurfaceCallback();
  virtual void LoadVTKSurfaceCallback();
  virtual void SaveSurfaceCallback();
  virtual void SaveSTLSurfaceCallback();
  virtual void SaveVTKSurfaceCallback();
  virtual void DeleteSurfaceCallback();

  vtkSetObjectMacro(BBoxList, vtkLinkedListWrapper);
  vtkSetObjectMacro(FEMeshList, vtkLinkedListWrapper);

  // save reference to the scene to be used for storage 
   void SetMRMLSceneForStorage(vtkMRMLScene* scene);
   
protected:
        vtkFESurfaceMRMLMenuGroup();
        ~vtkFESurfaceMRMLMenuGroup();
        vtkKWFileBrowserDialog *FileBrowserDialog;
        virtual void CreateWidget();
        vtkKWMimxSaveSTLSurfaceGroup *SaveSTLGroup;
        vtkKWMimxSaveVTKSurfaceGroup *SaveVTKGroup;
        vtkKWMimxDeleteObjectGroup   *DeleteObjectGroup;
        vtkMRMLScene                 *savedMRMLScene;
private:
  vtkFESurfaceMRMLMenuGroup(const vtkFESurfaceMRMLMenuGroup&); // Not implemented
  void operator=(const vtkFESurfaceMRMLMenuGroup&); // Not implemented
 };

#endif
