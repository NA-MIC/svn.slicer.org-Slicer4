/*=auto=========================================================================

Portions (c) Copyright 2005 Brigham and Women\"s Hospital (BWH) All Rights Reserved.

See COPYRIGHT.txt
or http://www.slicer.org/copyright/copyright.txt for details.

Program:   3D Slicer
Module:    $RCSfile: vtkMRMLDiffusionWeightedVolumeDisplayNode.cxx,v $
Date:      $Date: 2006/03/17 15:10:10 $
Version:   $Revision: 1.2 $

=========================================================================auto=*/

// MRML includes
#include "vtkMRMLDiffusionWeightedVolumeDisplayNode.h"

// VTK includes
#include <vtkImageAppendComponents.h>
#include <vtkImageData.h>
#include <vtkImageExtractComponents.h>
#include <vtkImageMapToWindowLevelColors.h>
#include <vtkImageThreshold.h>
#include <vtkObjectFactory.h>
#include <vtkVersion.h>

// STD includes
#include <sstream>

//------------------------------------------------------------------------------
vtkMRMLNodeNewMacro(vtkMRMLDiffusionWeightedVolumeDisplayNode);

//----------------------------------------------------------------------------
vtkMRMLDiffusionWeightedVolumeDisplayNode::vtkMRMLDiffusionWeightedVolumeDisplayNode()
{
  this->DiffusionComponent = 0;
  this->ExtractComponent = vtkImageExtractComponents::New();
  this->Threshold->SetInputConnection( this->ExtractComponent->GetOutputPort());
  this->MapToWindowLevelColors->SetInputConnection(
    this->ExtractComponent->GetOutputPort());
}

//----------------------------------------------------------------------------
vtkMRMLDiffusionWeightedVolumeDisplayNode::~vtkMRMLDiffusionWeightedVolumeDisplayNode()
{
  this->ExtractComponent->Delete();
}

//----------------------------------------------------------------------------
void vtkMRMLDiffusionWeightedVolumeDisplayNode::WriteXML(ostream& of, int nIndent)
{
  Superclass::WriteXML(of, nIndent);

  vtkIndent indent(nIndent);

  std::stringstream ss;
  ss << this->DiffusionComponent;
  of << indent << " diffusionComponent=\"" << ss.str() << "\"";
}

//----------------------------------------------------------------------------
void vtkMRMLDiffusionWeightedVolumeDisplayNode::ReadXMLAttributes(const char** atts)
{
  int disabledModify = this->StartModify();

  Superclass::ReadXMLAttributes(atts);

  const char* attName;
  const char* attValue;
  while (*atts != NULL)
    {
    attName = *(atts++);
    attValue = *(atts++);
    if (!strcmp(attName, "diffusionComponent"))
      {
      std::stringstream ss;
      ss << attValue;
      ss >> this->DiffusionComponent;
      }
    }
  this->EndModify(disabledModify);
}

//----------------------------------------------------------------------------
// Copy the node\"s attributes to this object.
// Does NOT copy: ID, FilePrefix, Name, VolumeID
void vtkMRMLDiffusionWeightedVolumeDisplayNode::Copy(vtkMRMLNode *anode)
{
  int disabledModify = this->StartModify();

  Superclass::Copy(anode);
  vtkMRMLDiffusionWeightedVolumeDisplayNode *node = (vtkMRMLDiffusionWeightedVolumeDisplayNode *) anode;

  this->SetDiffusionComponent(node->DiffusionComponent);

  this->EndModify(disabledModify);
}

//----------------------------------------------------------------------------
void vtkMRMLDiffusionWeightedVolumeDisplayNode::PrintSelf(ostream& os, vtkIndent indent)
{

  Superclass::PrintSelf(os,indent);

  os << indent << "Diffusion Component:   " << this->DiffusionComponent << "\n";

}

//----------------------------------------------------------------------------
#if (VTK_MAJOR_VERSION <= 5)
void vtkMRMLDiffusionWeightedVolumeDisplayNode
::SetInputToImageDataPipeline(vtkImageData *imageData)
{
  this->ExtractComponent->SetInput(imageData);
}
#else
void vtkMRMLDiffusionWeightedVolumeDisplayNode
::SetInputToImageDataPipeline(vtkAlgorithmOutput *imageDataPort)
{
  this->ExtractComponent->SetInputConnection(imageDataPort);
}
#endif

//----------------------------------------------------------------------------
vtkImageData* vtkMRMLDiffusionWeightedVolumeDisplayNode::GetInputImageData()
{
  return vtkImageData::SafeDownCast(this->ExtractComponent->GetInput());
}

//----------------------------------------------------------------------------
#if (VTK_MAJOR_VERSION <= 5)
vtkImageData* vtkMRMLDiffusionWeightedVolumeDisplayNode::GetOutputImageData()
{
  return this->AppendComponents->GetOutput();
}
#else
vtkAlgorithmOutput* vtkMRMLDiffusionWeightedVolumeDisplayNode::GetOutputImageDataPort()
{
  return this->AppendComponents->GetOutputPort();
}
#endif

//---------------------------------------------------------------------------
vtkImageData* vtkMRMLDiffusionWeightedVolumeDisplayNode::GetScalarImageData()
{
  return vtkImageData::SafeDownCast(this->ExtractComponent->GetOutput());
}

//----------------------------------------------------------------------------
void vtkMRMLDiffusionWeightedVolumeDisplayNode::UpdateImageDataPipeline()
{
  this->ExtractComponent->SetComponents(this->GetDiffusionComponent());
  this->Superclass::UpdateImageDataPipeline();
}
