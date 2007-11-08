/*=auto=========================================================================

Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

See Doc/copyright/copyright.txt
or http://www.slicer.org/copyright/copyright.txt for details.

Program:   3D Slicer
Module:    $RCSfile: vtkMRMLFiniteElementSurfaceNode.cxx,v $
Date:      $Date: 2006/03/17 15:10:10 $
Version:   $Revision: 1.2 $

=========================================================================auto=*/

#include <string>
#include <iostream>
#include <sstream>

#include "vtkObjectFactory.h"

#include "vtkMRMLFiniteElementSurfaceNode.h"
#include "vtkMRMLScene.h"

#include "vtkMimxSurfacePolyDataActor.h"

//------------------------------------------------------------------------------
vtkMRMLFiniteElementSurfaceNode* vtkMRMLFiniteElementSurfaceNode::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLFiniteElementSurfaceNode");
  if(ret)
    {
      return (vtkMRMLFiniteElementSurfaceNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLFiniteElementSurfaceNode;
}

//----------------------------------------------------------------------------

vtkMRMLNode* vtkMRMLFiniteElementSurfaceNode::CreateNodeInstance()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLFiniteElementSurfaceNode");
  if(ret)
    {
      return (vtkMRMLFiniteElementSurfaceNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLFiniteElementSurfaceNode;
}

//----------------------------------------------------------------------------
vtkMRMLFiniteElementSurfaceNode::vtkMRMLFiniteElementSurfaceNode()
{
  this->MimxActor = vtkMimxSurfacePolyDataActor::New();
}

// pass through the access to the Mimx actor we keep 
void vtkMRMLFiniteElementSurfaceNode::SetFilePath(const char *InputFilePath)
{
  this->MimxActor->SetFilePath(InputFilePath);
}

void vtkMRMLFiniteElementSurfaceNode::SetFileName(const char *InputFileName)
{
  this->MimxActor->SetFileName(InputFilePath);
}



//----------------------------------------------------------------------------
vtkMRMLFiniteElementSurfaceNode::~vtkMRMLFiniteElementSurfaceNode()
{
}

//----------------------------------------------------------------------------
void vtkMRMLFiniteElementSurfaceNode::WriteXML(ostream& of, int nIndent)
{
Superclass::WriteXML(of, nIndent);

  vtkIndent indent(nIndent);

  // write out an entry for each instance variable 
  
//  {
//    std::stringstream ss;
//    ss << this->Conductance;
//    of << indent << "Conductance='" << ss.str() << "' ";
//  }
  
}

//----------------------------------------------------------------------------
void vtkMRMLFiniteElementSurfaceNode::ReadXMLAttributes(const char** atts)
{

//  vtkMRMLNode::ReadXMLAttributes(atts);
//
//  const char* attName;
//  const char* attValue;
//  while (*atts != NULL) 
//    {
//    attName = *(atts++);
//    attValue = *(atts++);
//    if (!strcmp(attName, "Conductance")) 
//      {
//      std::stringstream ss;
//      ss << attValue;
//      ss >> this->Conductance;
//      }
//    else if (!strcmp(attName, "NumberOfIterations")) 
//      {
//      std::stringstream ss;
//      ss << attValue;
//      ss >> this->NumberOfIterations;
//      }
//    else if (!strcmp(attName, "TimeStep")) 
//      {
//      std::stringstream ss;
//      ss << attValue;
//      ss >> this->TimeStep;
//      }
//    else if (!strcmp(attName, "InputVolumeRef"))
//      {
//      std::stringstream ss;
//      ss << attValue;
//      ss >> this->InputVolumeRef;
//      }
//    else if (!strcmp(attName, "OutputVolumeRef"))
//      {
//      std::stringstream ss;
//      ss << attValue;
//      ss >> this->OutputVolumeRef;
//      }
//    }
}

//----------------------------------------------------------------------------
// Copy the node's attributes to this object.
// Does NOT copy: ID, FilePrefix, Name, VolumeID
void vtkMRMLFiniteElementSurfaceNode::Copy(vtkMRMLNode *anode)
{
  Superclass::Copy(anode);
  vtkMRMLFiniteElementSurfaceNode *node = (vtkMRMLFiniteElementSurfaceNode *) anode;

  // copy any variables here
  //this->SetConductance(node->Conductance);

}

//----------------------------------------------------------------------------
void vtkMRMLFiniteElementSurfaceNode::PrintSelf(ostream& os, vtkIndent indent)
{
  
  vtkMRMLNode::PrintSelf(os,indent);

  os << indent << "Finite Element Suface Node" << "\n";

}

