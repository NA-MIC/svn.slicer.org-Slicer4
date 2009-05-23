/*=auto=========================================================================

Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

See Doc/copyright/copyright.txt
or http://www.slicer.org/copyright/copyright.txt for details.

Program:   3D Slicer
Module:    $RCSfile: vtkMRMLFiniteElementMeshQualityDisplayNode.cxx,v $
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

#include "vtkMRMLFiniteElementMeshQualityDisplayNode.h"
#include "vtkMRMLScene.h"
#include "vtkMimxBoundingBoxSource.h"

//------------------------------------------------------------------------------
vtkMRMLFiniteElementMeshQualityDisplayNode* vtkMRMLFiniteElementMeshQualityDisplayNode::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLFiniteElementMeshQualityDisplayNode");
  if(ret)
    {
    return (vtkMRMLFiniteElementMeshQualityDisplayNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLFiniteElementMeshQualityDisplayNode;
}

//-----------------------------------------------------------------------------
vtkMRMLNode* vtkMRMLFiniteElementMeshQualityDisplayNode::CreateNodeInstance()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLFiniteElementMeshQualityDisplayNode");
  if(ret)
    {
    return (vtkMRMLFiniteElementMeshQualityDisplayNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLFiniteElementMeshQualityDisplayNode;
}


vtkMRMLFiniteElementMeshQualityDisplayNode::vtkMRMLFiniteElementMeshQualityDisplayNode()
{
//    this->ElementShrinkFactor = 1.0;
//     this->DataType = ACTOR_FE_MESH;
//     this->ElementSetName = NULL;
//     this->IsVisible = true;
//     this->ElementSetDisplayList.clear();
//     this->DisplayMode = vtkMimxMeshActor::DisplayMesh;
//     this->DisplayType = vtkMimxMeshActor::DisplaySurfaceAndOutline;
//     this->NumberOfElementSets = 0;
//     this->CuttingPlaneEnabled = false;


//     vtkDataSetMapper*  meshMapper = vtkDataSetMapper::New();
//     meshMapper->SetInputConnection(this->ShrinkFilter->GetOutputPort());
//     this->UnstructuredGridMapper = meshMapper;
//     this->UnstructuredGridMapper->SetScalarVisibility(0);
//     this->Actor = vtkActor::New();
//     this->Actor->SetMapper(this->UnstructuredGridMapper);
           
     // when the user selects wireframe mode, we will change the VTK pipeline
     // to employ this wireframe filter instead of the filled poly pipeline
     
     /* Setup the Pipeline for the Wireframe */
 //    this->OutlineGeometryFilter = vtkGeometryFilter::New();
 //    this->OutlineGeometryFilter->SetInput( this->UnstructuredGrid );

//     this->FeatureEdges = vtkFeatureEdges::New();
//     this->FeatureEdges->SetInput( this->OutlineGeometryFilter->GetOutput() );
//     this->FeatureEdges->BoundaryEdgesOn();
//     this->FeatureEdges->ManifoldEdgesOn();
//     this->FeatureEdges->FeatureEdgesOff();
//     this->FeatureEdges->ColoringOff();    
//     this->TubeFilter = vtkTubeFilter::New();
//     this->TubeFilter->SetInputConnection(this->FeatureEdges->GetOutputPort());
//     this->TubeFilter->SetRadius(0.03);
      
//     this->OutlineMapper = vtkPolyDataMapper::New();
//     this->OutlineMapper->SetInputConnection(  this->TubeFilter->GetOutputPort() );
//     this->OutlineMapper->SetScalarVisibility( 0 );
//
//     this->OutlineActor->GetProperty()->SetColor(0.0,0.0,0.0);
//     this->OutlineActor->GetProperty()->SetRepresentationToSurface();
//     this->OutlineActor->GetProperty()->SetAmbient(1);
//     this->OutlineActor->GetProperty()->SetSpecular(0);
//     this->OutlineActor->GetProperty()->SetDiffuse(0);
//     this->OutlineActor->GetProperty()->SetSpecularPower(0);
     //this->OutlineActor->GetProperty()->SetLineWidth(1.0);    
    
//     this->CuttingPlaneWidget = vtkPlaneWidget::New();
//     this->CuttingPlaneWidget->SetInput( this->UnstructuredGrid );
//     this->CuttingPlaneWidget->SetRepresentationToSurface();
//     this->CuttingPlaneWidget->GetPlaneProperty()->SetColor(0.2,0.2,0);
//     this->CuttingPlaneWidget->GetPlaneProperty()->SetOpacity(0.2);
//     this->CuttingPlaneWidget->GetSelectedPlaneProperty()->SetOpacity(0.2);  
//     this->CuttingPlaneWidget->SetHandleSize(0.02);
     
     // this geometry can be clipped by a clipping plane, so set up the VTK pipeline
     // with the extract geometry filter in place that will do the selection of cells
     
//     this->CuttingPlane = vtkPlane::New();     
//     this->ClipPlaneGeometryFilter = vtkExtractGeometry::New();
//     this->ClipPlaneGeometryFilter->SetInput( this->UnstructuredGrid );
//   
     this->SavedMeshQualityFilter = vtkMeshQuality::New();
     //vtkMeshQualityExtended* this->SavedMeshQualityFilter = vtkMeshQualityExtended::New();
     this->SavedShrinkFilter = vtkShrinkFilter::New();
     this->ShrinkFactor = 0.80;
}


vtkMRMLFiniteElementMeshQualityDisplayNode::~vtkMRMLFiniteElementMeshQualityDisplayNode()
{
  this->RemoveObservers ( vtkCommand::ModifiedEvent, this->MRMLCallbackCommand );
  this->GeometryFilter->Delete();
  this->ShrinkPolyData->Delete();
  this->SavedShrinkFilter->Delete();
}

void vtkMRMLFiniteElementMeshQualityDisplayNode::UpdatePolyDataPipeline()
{
   // set the type of metric to display here and the paramters for coloring, etc. 
   //this->ShrinkFactor = whatever-was-in-the-GUI
   //this->ShrinkPolyData->SetShrinkFactor(this->ShrinkFactor); 
}

//----------------------------------------------------------------------------
void vtkMRMLFiniteElementMeshQualityDisplayNode::SetUnstructuredGrid(vtkUnstructuredGrid *grid)
{
    // assign the filter to add mesh quality scalars to points & cells
    this->SavedMeshQualityFilter->SetInput(grid);
    //this->SavedMeshQualityFilter->SetHexQualityMeasureToJacobian();
    this->SavedMeshQualityFilter->SetHexQualityMeasureToEdgeRatio();
    this->SavedMeshQualityFilter->SaveCellQualityOn(); 

    // shrink the output because the mappers will remove interior detail otherwise
    
    this->SavedShrinkFilter->SetInput(this->SavedMeshQualityFilter->GetOutput());
    this->SavedShrinkFilter->SetShrinkFactor(this->ShrinkFactor);   
    this->GeometryFilter->SetInput(this->SavedShrinkFilter->GetOutput());

}


  

//----------------------------------------------------------------------------
void vtkMRMLFiniteElementMeshQualityDisplayNode::WriteXML(ostream& of, int nIndent)
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
void vtkMRMLFiniteElementMeshQualityDisplayNode::ReadXMLAttributes(const char** atts)
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
vtkPolyData* vtkMRMLFiniteElementMeshQualityDisplayNode::GetPolyData()
{
  vtkDebugMacro("MeshDisplayNode invoked");
//  vtkDataSetWriter *write = vtkDataSetWriter::New();
//  write->SetInput(this->ShrinkPolyData->GetOutput());
//  write->SetFileName("mesh-with-quality-from-display-node.vtk");
//  write->Write();
  return this->GeometryFilter->GetOutput();
}


//----------------------------------------------------------------------------
// Copy the node's attributes to this object.
// Does NOT copy: ID, FilePrefix, Name, ID
void vtkMRMLFiniteElementMeshQualityDisplayNode::Copy(vtkMRMLNode *anode)
{
  Superclass::Copy(anode);
  vtkMRMLFiniteElementMeshQualityDisplayNode *node = (vtkMRMLFiniteElementMeshQualityDisplayNode *) anode;
  this->SetShrinkFactor(node->ShrinkFactor);
}

//----------------------------------------------------------------------------
void vtkMRMLFiniteElementMeshQualityDisplayNode::PrintSelf(ostream& os, vtkIndent indent)
{
  //int idx;
  
  Superclass::PrintSelf(os,indent);
  os << indent << "ShrinkFactor:             " << this->ShrinkFactor << "\n";
}


//---------------------------------------------------------------------------
void vtkMRMLFiniteElementMeshQualityDisplayNode::ProcessMRMLEvents ( vtkObject *caller,
                                           unsigned long event, 
                                           void *callData )
{
  Superclass::ProcessMRMLEvents(caller, event, callData);
  return;
}
