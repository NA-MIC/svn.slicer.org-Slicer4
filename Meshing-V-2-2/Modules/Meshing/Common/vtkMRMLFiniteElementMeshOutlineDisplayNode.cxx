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
void vtkMRMLFiniteElementMeshOutlineDisplayNode::SetCuttingPlane(vtkPlane *plane)
{
    this->SavedCuttingPlane = plane;
    if (this->SavedMeshQualityRendering != NULL)
    {
       this->SavedMeshQualityRendering->SetCuttingPlaneFunction(plane);
    }
}


//----------------------------------------------------------------------------
vtkMRMLFiniteElementMeshOutlineDisplayNode::vtkMRMLFiniteElementMeshOutlineDisplayNode()
{
    this->SavedMeshQualityRendering = vtkMimxMeshQualityRendering::New();
    this->SavedCuttingPlane = NULL;
}



vtkMRMLFiniteElementMeshOutlineDisplayNode::~vtkMRMLFiniteElementMeshOutlineDisplayNode()
{
  this->RemoveObservers ( vtkCommand::ModifiedEvent, this->MRMLCallbackCommand );
}

//----------------------------------------------------------------------------
void vtkMRMLFiniteElementMeshOutlineDisplayNode::SetUnstructuredGrid(vtkUnstructuredGrid *grid)
{
    this->SavedMeshQualityRendering->InitializeFromExternalMesh(grid);
   // put in a null plane for now so we can instantiate the pipelines
    if (this->SavedCuttingPlane == NULL)
        this->SavedCuttingPlane = vtkPlane::New();
    this->SavedMeshQualityRendering->SetCuttingPlaneFunction(  this->SavedCuttingPlane);
    this->SavedMeshQualityRendering->SetShowClippedOutline(1);
    //this->SavedMeshQualityRendering->SetQualityMeasureToJacobian();
    this->SavedMeshQualityRendering->CalculateMeshQuality();
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
      return this->SavedMeshQualityRendering->GetOutlinePolygons();
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
    //this->SavedMeshQualityRendering->SetRadius(radius);
    cout << "outline display node: change radius" << endl;
}
