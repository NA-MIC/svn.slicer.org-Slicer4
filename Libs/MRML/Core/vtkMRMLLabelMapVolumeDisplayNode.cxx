/*=auto=========================================================================

Portions (c) Copyright 2005 Brigham and Women\"s Hospital (BWH) All Rights Reserved.

See COPYRIGHT.txt
or http://www.slicer.org/copyright/copyright.txt for details.

Program:   3D Slicer
Module:    $RCSfile: vtkMRMLVolumeDisplayNode.cxx,v $
Date:      $Date: 2006/03/17 15:10:10 $
Version:   $Revision: 1.2 $

=========================================================================auto=*/

#include "vtkMRMLLabelMapVolumeDisplayNode.h"
#include "vtkMRMLProceduralColorNode.h"
#include "vtkMRMLScene.h"

// VTK includes
#include <vtkImageData.h>
#include <vtkImageMapToColors.h>
#include <vtkLookupTable.h>
#include <vtkNew.h>
#include <vtkObjectFactory.h>
#include <vtkVersion.h>

// STD includes
#include <cassert>

//----------------------------------------------------------------------------
vtkMRMLNodeNewMacro(vtkMRMLLabelMapVolumeDisplayNode);

//----------------------------------------------------------------------------
vtkMRMLLabelMapVolumeDisplayNode::vtkMRMLLabelMapVolumeDisplayNode()
{
  this->MapToColors = vtkImageMapToColors::New();
  this->MapToColors->SetOutputFormatToRGBA();

  // set a thicker default slice intersection thickness for use when showing
  // the outline of the label map
  this->SliceIntersectionThickness = 3;
}

//----------------------------------------------------------------------------
vtkMRMLLabelMapVolumeDisplayNode::~vtkMRMLLabelMapVolumeDisplayNode()
{
   this->MapToColors->Delete();
}

//----------------------------------------------------------------------------
void vtkMRMLLabelMapVolumeDisplayNode::SetDefaultColorMap()
{
  // set up a default color node
  // TODO: figure out if can use vtkSlicerColorLogic's helper methods
  this->SetAndObserveColorNodeID("vtkMRMLColorTableNodeLabels");
}

//----------------------------------------------------------------------------
void vtkMRMLLabelMapVolumeDisplayNode::PrintSelf(ostream& os, vtkIndent indent)
{

  Superclass::PrintSelf(os,indent);
}

//---------------------------------------------------------------------------
void vtkMRMLLabelMapVolumeDisplayNode::ProcessMRMLEvents ( vtkObject *caller,
                                           unsigned long event,
                                           void *callData )
{
  Superclass::ProcessMRMLEvents(caller, event, callData);
}

//---------------------------------------------------------------------------
#if (VTK_MAJOR_VERSION <= 5)
void vtkMRMLLabelMapVolumeDisplayNode::SetInputImageData(vtkImageData *imageData)
{
  this->MapToColors->SetInput(imageData);
}
#else
void vtkMRMLLabelMapVolumeDisplayNode::SetInputImageDataConnection(vtkAlgorithmOutput *imageDataConnection)
{
  this->MapToColors->SetInputConnection(imageDataConnection);
}
#endif

//---------------------------------------------------------------------------
vtkImageData* vtkMRMLLabelMapVolumeDisplayNode::GetInputImageData()
{
  return vtkImageData::SafeDownCast(this->MapToColors->GetInput());
}

//---------------------------------------------------------------------------
#if (VTK_MAJOR_VERSION <= 5)
vtkImageData* vtkMRMLLabelMapVolumeDisplayNode::GetOutputImageData()
#else
vtkAlgorithmOutput* vtkMRMLLabelMapVolumeDisplayNode::GetOutputImageDataConnection()
#endif
{
  assert(!this->MapToColors->GetLookupTable() ||
         !this->MapToColors->GetLookupTable()->IsA("vtkLookupTable") ||
         vtkLookupTable::SafeDownCast(this->MapToColors->GetLookupTable())->GetNumberOfTableValues());
#if (VTK_MAJOR_VERSION <= 5)
  return this->MapToColors->GetOutput();
#else
  return this->MapToColors->GetOutputPort();
#endif
}

//---------------------------------------------------------------------------
void vtkMRMLLabelMapVolumeDisplayNode::UpdateImageDataPipeline()
{
  Superclass::UpdateImageDataPipeline();

  vtkScalarsToColors *lookupTable = NULL;
  if (this->GetColorNode())
    {
    lookupTable = this->GetColorNode()->GetLookupTable();
    if (lookupTable == NULL)
      {
      if (vtkMRMLProceduralColorNode::SafeDownCast(this->GetColorNode()) != NULL)
        {
        vtkDebugMacro("UpdateImageDataPipeline: getting a color transfer function");
        lookupTable = (vtkScalarsToColors*)(vtkMRMLProceduralColorNode::SafeDownCast(this->GetColorNode())->GetColorTransferFunction());
        }
      }
    }
  if (lookupTable == NULL && this->ColorNodeID != NULL)
    {
    // only complain if there's a scene set and that scene is not batch processing
    if (this->GetScene() && !this->GetScene()->IsBatchProcessing())
      {
      vtkWarningMacro(<< "vtkMRMLLabelMapVolumeDisplayNode: Warning, the color table node: "
                    << this->ColorNodeID << " can't be found");
      }
    }
  if (lookupTable && lookupTable->IsA("vtkLookupTable"))
    {
    // make a copy so that the range can be adjusted
    vtkNew<vtkLookupTable> lut;
    lut->DeepCopy(lookupTable);
    this->MapToColors->SetLookupTable(lut.GetPointer());
    // make sure that the table range matches number of colors for proper drawing
    // Tables are generally set up with the range set to 0-255 for non label map
    // volume scalar mapping, but for label maps, we want a 1:1 mapping that doesn't get scaled.
    if ((lut->GetTableRange()[1] - lut->GetTableRange()[0] + 1)
        != lut->GetNumberOfTableValues())
      {
      lut->SetTableRange(0,lut->GetNumberOfTableValues() - 1);
      }
    }
  else
    {
    this->MapToColors->SetLookupTable(lookupTable);
    }
  // if there is no point, the mapping will fail (not sure)
  assert(!lookupTable || !vtkLookupTable::SafeDownCast(lookupTable) ||
         vtkLookupTable::SafeDownCast(lookupTable)->GetNumberOfTableValues());
}
