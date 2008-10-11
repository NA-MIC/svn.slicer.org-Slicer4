/*=========================================================================

Program:   MIMX Meshing Toolkit
Module:    $RCSfile: vtkKWMimxApplyFEMeshMaterialPropertiesFromImageGroup.h,v $
Language:  C++
Date:      $Date: 2008/04/24 14:08:40 $
Version:   $Revision: 1.8 $

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
// .NAME vtkKWMimxApplyFEMeshMaterialPropertiesFromImageGroup - a tabbed notebook of UI pages
// .SECTION Description
// The class is derived from vtkKWMimxGroupBase. It contains 5 pages 1) Image
// 2) Surface 3) Building Block 4) F E Mesh 5) Mesh quality. Each page inturn
// contains a list of options specific to a page.

#ifndef __vtkKWMimxApplyFEMeshMaterialPropertiesFromImageGroup_h
#define __vtkKWMimxApplyFEMeshMaterialPropertiesFromImageGroup_h

#include "vtkKWMimxGroupBase.h"
#include "vtkKWMimxFEMeshMenuGroup.h"
#include "vtkKWMimxMainWindow.h"
#include "vtkKWMimxSurfaceMenuGroup.h"

class vtkKWComboBoxWithLabel;
class vtkIntArray;
class vtkKWEntryWithLabel;
class vtkKWCheckButtonWithLabel;
class vtkKWMenuButtonWithLabel;
class vtkKWFrameWithLabel;


class vtkKWMimxApplyFEMeshMaterialPropertiesFromImageGroup : public vtkKWMimxGroupBase
{
public:
  static vtkKWMimxApplyFEMeshMaterialPropertiesFromImageGroup* New();
  vtkTypeRevisionMacro(vtkKWMimxApplyFEMeshMaterialPropertiesFromImageGroup,vtkKWMimxGroupBase);
  void PrintSelf(ostream& os, vtkIndent indent);
  virtual void Update();
  virtual void UpdateEnableState();
  void ApplyFEMeshMaterialPropertiesFromImageDoneCallback();
  int ApplyFEMeshMaterialPropertiesFromImageApplyCallback();
  void ApplyFEMeshMaterialPropertiesFromImageCancelCallback();
  void UpdateObjectLists();
  void FEMeshSelectionChangedCallback(const char *Selection);
  void ElementSetChangedCallback(const char *Selection);
  
  void ViewMaterialPropertyCallback( int mode );
  int ClippingPlaneCallback(int mode);
  void ViewPropertyLegendCallback( int mode );

protected:
        vtkKWMimxApplyFEMeshMaterialPropertiesFromImageGroup();
        ~vtkKWMimxApplyFEMeshMaterialPropertiesFromImageGroup();
        virtual void CreateWidget();
  vtkKWComboBoxWithLabel *ImageListComboBox;
  vtkKWComboBoxWithLabel *FEMeshListComboBox;
  vtkKWComboBoxWithLabel *ElementSetComboBox;
  vtkKWEntryWithLabel *PoissonsRatioEntry;
  vtkKWFrameWithLabel *ViewFrame;
  vtkKWCheckButtonWithLabel *ViewPropertyButton;
  vtkKWCheckButtonWithLabel *ViewLegendButton;
  vtkKWMenuButtonWithLabel *ClippingPlaneMenuButton;
private:
  vtkKWMimxApplyFEMeshMaterialPropertiesFromImageGroup(const vtkKWMimxApplyFEMeshMaterialPropertiesFromImageGroup&); // Not implemented
  void operator=(const vtkKWMimxApplyFEMeshMaterialPropertiesFromImageGroup&); // Not implemented
  
  char meshName[64];
  char elementSetName[64];
  
 };

#endif

