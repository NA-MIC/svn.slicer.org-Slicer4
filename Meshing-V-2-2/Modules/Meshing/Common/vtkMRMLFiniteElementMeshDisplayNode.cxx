/*=auto=========================================================================

Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

See Doc/copyright/copyright.txt
or http://www.slicer.org/copyright/copyright.txt for details.

Program:   3D Slicer
Module:    $RCSfile: vtkMRMLFiniteElementMeshDisplayNode.cxx,v $
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

#include "vtkPlane.h"

#include "vtkMRMLFiniteElementMeshDisplayNode.h"
#include "vtkMRMLScene.h"
#include "vtkMimxBoundingBoxSource.h"

// for debugging purposes
#include "vtkUnstructuredGridWriter.h"


//------------------------------------------------------------------------------
vtkMRMLFiniteElementMeshDisplayNode* vtkMRMLFiniteElementMeshDisplayNode::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLFiniteElementMeshDisplayNode");
  if(ret)
    {
    return (vtkMRMLFiniteElementMeshDisplayNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLFiniteElementMeshDisplayNode;
}

//-----------------------------------------------------------------------------
vtkMRMLNode* vtkMRMLFiniteElementMeshDisplayNode::CreateNodeInstance()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLFiniteElementMeshDisplayNode");
  if(ret)
    {
    return (vtkMRMLFiniteElementMeshDisplayNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLFiniteElementMeshDisplayNode;
}


void vtkMRMLFiniteElementMeshDisplayNode::UpdatePolyDataPipeline()
{
    this->SavedMeshQualityRendering->UpdatePipeline();
    this->SetActiveScalarName("Jacobian");
       this->ScalarVisibilityOn( );
       cout << "MRMLFEMeshDisplayNode: (updatePipeline) scalar viz on" << endl;
}

//----------------------------------------------------------------------------
vtkMRMLFiniteElementMeshDisplayNode::vtkMRMLFiniteElementMeshDisplayNode()
{
  this->SavedMeshQualityRendering = vtkMimxMeshQualityRendering::New();
  this->SavedCuttingPlane = vtkPlane::New();
  cout << "MRMLFEMeshDisplayNode: (constructor)" << endl;

}



vtkMRMLFiniteElementMeshDisplayNode::~vtkMRMLFiniteElementMeshDisplayNode()
{
  this->RemoveObservers ( vtkCommand::ModifiedEvent, this->MRMLCallbackCommand );
 if (SavedMeshQualityRendering != NULL)   this->SavedMeshQualityRendering->Delete();

}

//----------------------------------------------------------------------------
void vtkMRMLFiniteElementMeshDisplayNode::SetUnstructuredGrid(vtkUnstructuredGrid *grid)
{
//    // assign the filter to add mesh quality scalars to points & cells


    this->SavedMeshQualityRendering->InitializeFromExternalMesh(grid);
   // put in a null plane for now so we can instantiate the pipelines
    if (this->SavedCuttingPlane == NULL)
        this->SavedCuttingPlane = vtkPlane::New();
    this->SavedMeshQualityRendering->SetCuttingPlaneFunction(  this->SavedCuttingPlane);
    this->SavedMeshQualityRendering->SetShowFilledElements(1);
    this->SavedMeshQualityRendering->SetThresholdValue(1.0);
    this->SavedMeshQualityRendering->SetQualityMeasureToJacobian();
    this->SavedMeshQualityRendering->CalculateMeshQuality();
    this->SetActiveScalarName("Jacobian");
    this->ScalarVisibilityOn( );
}

//----------------------------------------------------------------------------
void vtkMRMLFiniteElementMeshDisplayNode::SetCuttingPlane(vtkPlane *plane)
{
    this->SavedCuttingPlane = plane;
    if (this->SavedMeshQualityRendering != NULL)
    {
       this->SavedMeshQualityRendering->SetCuttingPlaneFunction(plane);
    }
}

//----------------------------------------------------------------------------
void vtkMRMLFiniteElementMeshDisplayNode::SetElementSize(double shrink)
{
         if (SavedMeshQualityRendering != NULL)   this->SavedMeshQualityRendering->SetElementShrinkFactor(shrink);
}

//----------------------------------------------------------------------------
void vtkMRMLFiniteElementMeshDisplayNode::WriteXML(ostream& of, int nIndent)
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
void vtkMRMLFiniteElementMeshDisplayNode::ReadXMLAttributes(const char** atts)
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
vtkPolyData* vtkMRMLFiniteElementMeshDisplayNode::GetPolyData()
{
  vtkDebugMacro("MeshDisplayNode invoked");
  return this->SavedMeshQualityRendering->GetMeshPolygons();
}


//----------------------------------------------------------------------------
// Copy the node's attributes to this object.
// Does NOT copy: ID, FilePrefix, Name, ID
void vtkMRMLFiniteElementMeshDisplayNode::Copy(vtkMRMLNode *anode)
{
  Superclass::Copy(anode);
  vtkMRMLFiniteElementMeshDisplayNode *node = (vtkMRMLFiniteElementMeshDisplayNode *) anode;
  this->SetShrinkFactor(node->ShrinkFactor);
}

//----------------------------------------------------------------------------
void vtkMRMLFiniteElementMeshDisplayNode::PrintSelf(ostream& os, vtkIndent indent)
{
  //int idx;

  Superclass::PrintSelf(os,indent);
  os << indent << "ShrinkFactor:             " << this->ShrinkFactor << "\n";
}


//---------------------------------------------------------------------------
void vtkMRMLFiniteElementMeshDisplayNode::ProcessMRMLEvents ( vtkObject *caller,
                                           unsigned long event,
                                           void *callData )
{
  Superclass::ProcessMRMLEvents(caller, event, callData);
  return;
}
