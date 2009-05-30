/*=auto=========================================================================

Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

See Doc/copyright/copyright.txt
or http://www.slicer.org/copyright/copyright.txt for details.

Program:   3D Slicer
Module:    $RCSfile: vtkMRMLFiniteElementMeshOutlineDisplayNode.cxx,v $
Date:      $Date: 2006/03/03 22:26:39 $
Version:   $Revision: 1.3 $

=========================================================================auto=*/
#include <string>
#include <iostream>
#include <sstream>

#include "vtkObjectFactory.h"
#include "vtkCallbackCommand.h"
#include "vtkMeshQuality.h"
#include "vtkDataSetWriter.h"
#include "vtkShrinkFilter.h"
#include "vtkFeatureEdges.h"

#include "vtkMRMLFiniteElementMeshOutlineDisplayNode.h"
#include "vtkMRMLScene.h"
#include "vtkMimxBoundingBoxSource.h"

//------------------------------------------------------------------------------
vtkMRMLFiniteElementMeshOutlineDisplayNode* vtkMRMLFiniteElementMeshOutlineDisplayNode::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLFiniteElementMeshOutlineDisplayNode");
  if(ret)
    {
    return (vtkMRMLFiniteElementMeshOutlineDisplayNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLFiniteElementMeshOutlineDisplayNode;
}

//-----------------------------------------------------------------------------
vtkMRMLNode* vtkMRMLFiniteElementMeshOutlineDisplayNode::CreateNodeInstance()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLFiniteElementMeshOutlineDisplayNode");
  if(ret)
    {
    return (vtkMRMLFiniteElementMeshOutlineDisplayNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLFiniteElementMeshOutlineDisplayNode;
}


void vtkMRMLFiniteElementMeshOutlineDisplayNode::UpdatePolyDataPipeline()
{
   // set the type of metric to display here and the paramters for coloring, etc.
   //this->ShrinkFactor = whatever-was-in-the-GUI
   //this->ShrinkPolyData->SetShrinkFactor(this->ShrinkFactor);
}

//----------------------------------------------------------------------------
vtkMRMLFiniteElementMeshOutlineDisplayNode::vtkMRMLFiniteElementMeshOutlineDisplayNode()
{
  this->SavedMeshQualityFilter = vtkMeshQuality::New();
  //vtkMeshQualityExtended* this->SavedMeshQualityFilter = vtkMeshQualityExtended::New();
  this->SavedShrinkFilter = vtkShrinkFilter::New();
  this->ShrinkFactor = 0.80;

  this->FeatureEdges = vtkFeatureEdges::New();
  this->TubeFilter = vtkTubeFilter::New();

}



vtkMRMLFiniteElementMeshOutlineDisplayNode::~vtkMRMLFiniteElementMeshOutlineDisplayNode()
{
  this->RemoveObservers ( vtkCommand::ModifiedEvent, this->MRMLCallbackCommand );
  this->GeometryFilter->Delete();
  this->ShrinkPolyData->Delete();
  this->SavedShrinkFilter->Delete();
}

//----------------------------------------------------------------------------
void vtkMRMLFiniteElementMeshOutlineDisplayNode::SetUnstructuredGrid(vtkUnstructuredGrid *grid)
{

    // assign the filter to add mesh quality scalars to points & cells
    this->SavedMeshQualityFilter->SetInput(grid);
    //this->SavedMeshQualityFilter->SetHexQualityMeasureToJacobian();
    this->SavedMeshQualityFilter->SetHexQualityMeasureToEdgeRatio();
    this->SavedMeshQualityFilter->SaveCellQualityOn();

    // shrink the output because the mappers will remove interior detail otherwise

    this->SavedShrinkFilter->SetInput(this->SavedMeshQualityFilter->GetOutput());
    this->SavedShrinkFilter->SetShrinkFactor(this->ShrinkFactor);

    // *** instead of using shrink output, use the original grid to avoid
    // getting all internal nodes
    //this->GeometryFilter->SetInput(this->SavedShrinkFilter->GetOutput());
    this->GeometryFilter->SetInput(grid);

    this->FeatureEdges = vtkFeatureEdges::New();
    this->FeatureEdges->SetInput( this->GeometryFilter->GetOutput() );
    this->FeatureEdges->BoundaryEdgesOn();
    this->FeatureEdges->ManifoldEdgesOn();
    this->FeatureEdges->FeatureEdgesOff();
    this->FeatureEdges->ColoringOff();

    this->TubeFilter = vtkTubeFilter::New();
    this->TubeFilter->SetInputConnection(this->FeatureEdges->GetOutputPort());
    this->TubeFilter->SetRadius(0.06);


}




//----------------------------------------------------------------------------
void vtkMRMLFiniteElementMeshOutlineDisplayNode::WriteXML(ostream& of, int nIndent)
{
  // Write all attributes not equal to their defaults

  Superclass::WriteXML(of, nIndent);
  of << " ";
   vtkIndent indent(nIndent);
//   {
//     std::stringstream ss;
//     ss << this->actor->GetDataType();
//     of << indent << " savedVisibilityState =\"" << this->savedVisibilityState << "\"";
//   }
}



//----------------------------------------------------------------------------
void vtkMRMLFiniteElementMeshOutlineDisplayNode::ReadXMLAttributes(const char** atts)
{

  Superclass::ReadXMLAttributes(atts);

  const char* attName;
  const char* attValue;
  //int intAttribute;

  while (*atts != NULL)
    {
    attName = *(atts++);
    attValue = *(atts++);


    }
}

// declare a rendering pipeline for bblock data in this class
vtkPolyData* vtkMRMLFiniteElementMeshOutlineDisplayNode::GetPolyData()
{
  vtkDebugMacro("MeshOutlineDisplayNode invoked");
//  vtkDataSetWriter *write = vtkDataSetWriter::New();
//  write->SetInput(this->ShrinkPolyData->GetOutput());
//  write->SetFileName("mesh-with-quality-from-display-node.vtk");
//  write->Write();
  return this->TubeFilter->GetOutput();
}


//----------------------------------------------------------------------------
// Copy the node's attributes to this object.
// Does NOT copy: ID, FilePrefix, Name, ID
void vtkMRMLFiniteElementMeshOutlineDisplayNode::Copy(vtkMRMLNode *anode)
{
  Superclass::Copy(anode);
  vtkMRMLFiniteElementMeshOutlineDisplayNode *node = (vtkMRMLFiniteElementMeshOutlineDisplayNode *) anode;
  this->SetShrinkFactor(node->ShrinkFactor);
}

//----------------------------------------------------------------------------
void vtkMRMLFiniteElementMeshOutlineDisplayNode::PrintSelf(ostream& os, vtkIndent indent)
{
  //int idx;

  Superclass::PrintSelf(os,indent);
  os << indent << "ShrinkFactor:             " << this->ShrinkFactor << "\n";
}


//---------------------------------------------------------------------------
void vtkMRMLFiniteElementMeshOutlineDisplayNode::ProcessMRMLEvents ( vtkObject *caller,
                                           unsigned long event,
                                           void *callData )
{
  Superclass::ProcessMRMLEvents(caller, event, callData);
  return;
}

void vtkMRMLFiniteElementMeshOutlineDisplayNode::SetRadius (float radius)
{
    this->TubeFilter->SetRadius(radius);
}
