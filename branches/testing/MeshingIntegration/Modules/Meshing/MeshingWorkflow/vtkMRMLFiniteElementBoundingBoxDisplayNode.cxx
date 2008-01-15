/*=auto=========================================================================

Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

See Doc/copyright/copyright.txt
or http://www.slicer.org/copyright/copyright.txt for details.

Program:   3D Slicer
Module:    $RCSfile: vtkMRMLFiniteElementBoundingBoxDisplayNode.cxx,v $
Date:      $Date: 2006/03/03 22:26:39 $
Version:   $Revision: 1.3 $

=========================================================================auto=*/
#include <string>
#include <iostream>
#include <sstream>

#include "vtkObjectFactory.h"
#include "vtkCallbackCommand.h"

#include "vtkMRMLFiniteElementBoundingBoxDisplayNode.h"
#include "vtkMRMLScene.h"
#include "vtkMimxBoundingBoxSource.h"

//------------------------------------------------------------------------------
vtkMRMLFiniteElementBoundingBoxDisplayNode* vtkMRMLFiniteElementBoundingBoxDisplayNode::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLFiniteElementBoundingBoxDisplayNode");
  if(ret)
    {
    return (vtkMRMLFiniteElementBoundingBoxDisplayNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLFiniteElementBoundingBoxDisplayNode;
}

//-----------------------------------------------------------------------------
vtkMRMLNode* vtkMRMLFiniteElementBoundingBoxDisplayNode::CreateNodeInstance()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLFiniteElementBoundingBoxDisplayNode");
  if(ret)
    {
    return (vtkMRMLFiniteElementBoundingBoxDisplayNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLFiniteElementBoundingBoxDisplayNode;
}


//----------------------------------------------------------------------------
vtkMRMLFiniteElementBoundingBoxDisplayNode::vtkMRMLFiniteElementBoundingBoxDisplayNode()
{
  
  
  this->ShrinkPolyData->SetInput( this->GeometryFilter->GetOutput());
  this->ShrinkFactor = 0.5;
  this->ShrinkPolyData->SetShrinkFactor(this->ShrinkFactor);
}


//----------------------------------------------------------------------------
vtkMRMLFiniteElementBoundingBoxDisplayNode::~vtkMRMLFiniteElementBoundingBoxDisplayNode()
{
  this->RemoveObservers ( vtkCommand::ModifiedEvent, this->MRMLCallbackCommand );
  this->GeometryFilter->Delete();
  this->ShrinkPolyData->Delete();
}

//----------------------------------------------------------------------------
void vtkMRMLFiniteElementBoundingBoxDisplayNode::WriteXML(ostream& of, int nIndent)
{
  // Write all attributes not equal to their defaults
  
  Superclass::WriteXML(of, nIndent);

  vtkIndent indent(nIndent);

  of << indent << " shrinkFactor =\"" << this->ShrinkFactor << "\"";
}



//----------------------------------------------------------------------------
void vtkMRMLFiniteElementBoundingBoxDisplayNode::ReadXMLAttributes(const char** atts)
{

  Superclass::ReadXMLAttributes(atts);

  const char* attName;
  const char* attValue;
  while (*atts != NULL) 
    {
    attName = *(atts++);
    attValue = *(atts++);

    if (!strcmp(attName, "shrinkFactor")) 
      {
      std::stringstream ss;
      ss << attValue;
      ss >> ShrinkFactor;
      }
    }  
}


//----------------------------------------------------------------------------
// Copy the node's attributes to this object.
// Does NOT copy: ID, FilePrefix, Name, ID
void vtkMRMLFiniteElementBoundingBoxDisplayNode::Copy(vtkMRMLNode *anode)
{
  Superclass::Copy(anode);
  vtkMRMLFiniteElementBoundingBoxDisplayNode *node = (vtkMRMLFiniteElementBoundingBoxDisplayNode *) anode;
  this->SetShrinkFactor(node->ShrinkFactor);
}

//----------------------------------------------------------------------------
void vtkMRMLFiniteElementBoundingBoxDisplayNode::PrintSelf(ostream& os, vtkIndent indent)
{
  //int idx;
  
  Superclass::PrintSelf(os,indent);
  os << indent << "ShrinkFactor:             " << this->ShrinkFactor << "\n";
}


//---------------------------------------------------------------------------
void vtkMRMLFiniteElementBoundingBoxDisplayNode::ProcessMRMLEvents ( vtkObject *caller,
                                           unsigned long event, 
                                           void *callData )
{
  Superclass::ProcessMRMLEvents(caller, event, callData);
  return;
}
