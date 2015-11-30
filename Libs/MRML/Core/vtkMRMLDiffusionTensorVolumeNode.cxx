/*=auto=========================================================================

Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

See COPYRIGHT.txt
or http://www.slicer.org/copyright/copyright.txt for details.

Program:   3D Slicer
Module:    $RCSfile: vtkMRMLVolumeNode.cxx,v $
Date:      $Date: 2006/03/17 17:01:53 $
Version:   $Revision: 1.14 $

=========================================================================auto=*/


// MRML includes
#include "vtkMRMLDiffusionTensorVolumeDisplayNode.h"
#include "vtkMRMLDiffusionTensorVolumeNode.h"
#include "vtkMRMLNRRDStorageNode.h"
#include "vtkMRMLScene.h"

// VTK includes
#include <vtkNew.h>
#include <vtkObjectFactory.h>

//------------------------------------------------------------------------------
vtkMRMLNodeNewMacro(vtkMRMLDiffusionTensorVolumeNode);

//----------------------------------------------------------------------------
vtkMRMLDiffusionTensorVolumeNode::vtkMRMLDiffusionTensorVolumeNode()
{
  this->Order = 2; //Second order Tensor
}

//----------------------------------------------------------------------------
void vtkMRMLDiffusionTensorVolumeNode::SetAndObserveDisplayNodeID(const char *displayNodeID)
{
  this->Superclass::SetAndObserveDisplayNodeID(displayNodeID);
  // Make sure the node added is a DiffusionTensorVolumeDisplayNode
  vtkMRMLNode* displayNode =  this->GetDisplayNode();
  if (displayNode && !vtkMRMLDiffusionTensorVolumeDisplayNode::SafeDownCast(displayNode))
    {
    vtkWarningMacro("SetAndObserveDisplayNodeID: The node to display "
                    << displayNodeID << " can NOT display diffusion tensors");
    }
}

//----------------------------------------------------------------------------
vtkMRMLDiffusionTensorVolumeNode::~vtkMRMLDiffusionTensorVolumeNode()
{
}

//----------------------------------------------------------------------------
void vtkMRMLDiffusionTensorVolumeNode::PrintSelf(ostream& os, vtkIndent indent)
{
  Superclass::PrintSelf(os,indent);
}

//----------------------------------------------------------------------------
vtkMRMLDiffusionTensorVolumeDisplayNode* vtkMRMLDiffusionTensorVolumeNode
::GetDiffusionTensorVolumeDisplayNode()
{
  return vtkMRMLDiffusionTensorVolumeDisplayNode::SafeDownCast(this->GetDisplayNode());
}

//----------------------------------------------------------------------------
vtkMRMLStorageNode* vtkMRMLDiffusionTensorVolumeNode::CreateDefaultStorageNode()
{
  return vtkMRMLNRRDStorageNode::New();
}

//----------------------------------------------------------------------------
void vtkMRMLDiffusionTensorVolumeNode::CreateDefaultDisplayNodes()
{
  if (vtkMRMLDiffusionTensorVolumeDisplayNode::SafeDownCast(this->GetDisplayNode())!=NULL)
    {
    // display node already exists
    return;
    }
  if (this->GetScene()==NULL)
    {
    vtkErrorMacro("vtkMRMLDiffusionTensorVolumeNode::CreateDefaultDisplayNodes failed: scene is invalid");
    return;
    }
  vtkNew<vtkMRMLDiffusionTensorVolumeDisplayNode> dispNode;
  this->GetScene()->AddNode(dispNode.GetPointer());
  dispNode->SetDefaultColorMap();
  this->SetAndObserveDisplayNodeID(dispNode->GetID());
  // add slice display nodes
  dispNode->AddSliceGlyphDisplayNodes( this );
}
