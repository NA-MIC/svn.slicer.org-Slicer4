/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See COPYRIGHT.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkMRMLVolumeNode.h,v $
  Date:      $Date: 2006/03/19 17:12:29 $
  Version:   $Revision: 1.13 $

=========================================================================auto=*/

#ifndef __vtkMRMLDiffusionTensorVolumeNode_h
#define __vtkMRMLDiffusionTensorVolumeNode_h

#include "vtkMRMLDiffusionImageVolumeNode.h"

class vtkMRMLDiffusionTensorVolumeDisplayNode;

/// \brief MRML node for representing diffusion weighted MRI volume.
///
/// Diffusion Weigthed Volume nodes describe data sets that encode diffusion weigthed
/// images. These images are the basis for computing the diffusion tensor.
/// The node is a container for the neccesary information to interpert DW images:
/// 1. Gradient information.
/// 2. B value for each gradient.
/// 3. Measurement frame that relates the coordinate system where the gradients are given
///  to RAS.
class VTK_MRML_EXPORT vtkMRMLDiffusionTensorVolumeNode : public vtkMRMLDiffusionImageVolumeNode
{
  public:
  static vtkMRMLDiffusionTensorVolumeNode *New();
  vtkTypeMacro(vtkMRMLDiffusionTensorVolumeNode,vtkMRMLDiffusionImageVolumeNode);
  void PrintSelf(ostream& os, vtkIndent indent);

  virtual vtkMRMLNode* CreateNodeInstance();

  /// Get node XML tag name (like Volume, Model)
  virtual const char* GetNodeTagName() { return "DiffusionTensorVolume"; }

  /// Associated volume display MRML node
  virtual void SetAndObserveDisplayNodeID(const char *DisplayNodeID);

  /// Associated display MRML node
  virtual vtkMRMLDiffusionTensorVolumeDisplayNode* GetDiffusionTensorVolumeDisplayNode();

  /// Create default storage node or NULL if does not have one
  virtual vtkMRMLStorageNode* CreateDefaultStorageNode();

  /// Create and observe default display node
  virtual void CreateDefaultDisplayNodes();

protected:
  vtkMRMLDiffusionTensorVolumeNode();
  ~vtkMRMLDiffusionTensorVolumeNode();

  vtkMRMLDiffusionTensorVolumeNode(const vtkMRMLDiffusionTensorVolumeNode&);
  void operator=(const vtkMRMLDiffusionTensorVolumeNode&);

};

#endif
