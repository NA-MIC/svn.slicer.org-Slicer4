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

class VTK_BOUNDINGBOX_EXPORT vtkFESurfaceMRMLMenuGroup : public vtkKWMimxSurfaceMenuGroup
{
public:
  static vtkFESurfaceMRMLMenuGroup* New();
  vtkTypeRevisionMacro(vtkFESurfaceMRMLMenuGroup,vtkKWMimxMainMenuGroup);
  void PrintSelf(ostream& os, vtkIndent indent);
  virtual void Update();
  virtual void UpdateEnableState();
  void SurfaceMenuCallback();
  void LoadSurfaceCallback();
  void LoadSTLSurfaceCallback();
  void LoadVTKSurfaceCallback();
  void SaveSurfaceCallback();
  void SaveSTLSurfaceCallback();
  void SaveVTKSurfaceCallback();
  void DeleteSurfaceCallback();

  vtkSetObjectMacro(BBoxList, vtkLinkedListWrapper);
  vtkSetObjectMacro(FEMeshList, vtkLinkedListWrapper);

protected:
        vtkFESurfaceMRMLMenuGroup();
        ~vtkFESurfaceMRMLMenuGroup();
        vtkKWFileBrowserDialog *FileBrowserDialog;
        virtual void CreateWidget();
        vtkKWMimxSaveSTLSurfaceGroup *SaveSTLGroup;
        vtkKWMimxSaveVTKSurfaceGroup *SaveVTKGroup;
        vtkKWMimxDeleteObjectGroup *DeleteObjectGroup;
private:
  vtkFESurfaceMRMLMenuGroup(const vtkFESurfaceMRMLMenuGroup&); // Not implemented
  void operator=(const vtkFESurfaceMRMLMenuGroup&); // Not implemented
 };

#endif
