/*=auto=========================================================================

Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

See Doc/copyright/copyright.txt
or http://www.slicer.org/copyright/copyright.txt for details.

Program:   3D Slicer
Module:    $RCSfile: vtkMRMLMeshingWorkflowNode.cxx,v $
Date:      $Date: 2006/03/17 15:10:10 $
Version:   $Revision: 1.2 $

=========================================================================auto=*/

#include <string>
#include <iostream>
#include <sstream>

#include "vtkObjectFactory.h"

#include "vtkMRMLMeshingWorkflowNode.h"
#include "vtkMRMLScene.h"


//------------------------------------------------------------------------------
vtkMRMLMeshingWorkflowNode* vtkMRMLMeshingWorkflowNode::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLMeshingWorkflowNode");
  if(ret)
    {
      return (vtkMRMLMeshingWorkflowNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLMeshingWorkflowNode;
}

//----------------------------------------------------------------------------

vtkMRMLNode* vtkMRMLMeshingWorkflowNode::CreateNodeInstance()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLMeshingWorkflowNode");
  if(ret)
    {
      return (vtkMRMLMeshingWorkflowNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLMeshingWorkflowNode;
}

//----------------------------------------------------------------------------
vtkMRMLMeshingWorkflowNode::vtkMRMLMeshingWorkflowNode()
{
   this->Conductance = 1.0;
   this->NumberOfIterations = 1;
   this->TimeStep = 0.1;
   this->InputVolumeRef = NULL;
   this->OutputVolumeRef = NULL;
}

//----------------------------------------------------------------------------
vtkMRMLMeshingWorkflowNode::~vtkMRMLMeshingWorkflowNode()
{
}

//----------------------------------------------------------------------------
void vtkMRMLMeshingWorkflowNode::WriteXML(ostream& of, int nIndent)
{
Superclass::WriteXML(of, nIndent);

  vtkIndent indent(nIndent);

  {
    std::stringstream ss;
    ss << this->Conductance;
    of << indent << "Conductance='" << ss.str() << "' ";
  }
  {
    std::stringstream ss;
    ss << this->NumberOfIterations;
    of << indent << "NumberOfIterations='" << ss.str() << "' ";
  }
  {
    std::stringstream ss;
    ss << this->TimeStep;
    of << indent << "TimeStep='" << ss.str() << "' ";
  }
  {
    std::stringstream ss;
    ss << this->InputVolumeRef;
    of << indent << "InputVolumeRef='" << ss.str() << "' ";
  }
  {
    std::stringstream ss;
    ss << this->OutputVolumeRef;
    of << indent << "OutputVolumeRef='" << ss.str() << "' ";
  }
}

//----------------------------------------------------------------------------
void vtkMRMLMeshingWorkflowNode::ReadXMLAttributes(const char** atts)
{

  vtkMRMLNode::ReadXMLAttributes(atts);

  const char* attName;
  const char* attValue;
  while (*atts != NULL) 
    {
    attName = *(atts++);
    attValue = *(atts++);
    if (!strcmp(attName, "Conductance")) 
      {
      std::stringstream ss;
      ss << attValue;
      ss >> this->Conductance;
      }
    else if (!strcmp(attName, "NumberOfIterations")) 
      {
      std::stringstream ss;
      ss << attValue;
      ss >> this->NumberOfIterations;
      }
    else if (!strcmp(attName, "TimeStep")) 
      {
      std::stringstream ss;
      ss << attValue;
      ss >> this->TimeStep;
      }
    else if (!strcmp(attName, "InputVolumeRef"))
      {
      std::stringstream ss;
      ss << attValue;
      ss >> this->InputVolumeRef;
      }
    else if (!strcmp(attName, "OutputVolumeRef"))
      {
      std::stringstream ss;
      ss << attValue;
      ss >> this->OutputVolumeRef;
      }
    }
}

//----------------------------------------------------------------------------
// Copy the node's attributes to this object.
// Does NOT copy: ID, FilePrefix, Name, VolumeID
void vtkMRMLMeshingWorkflowNode::Copy(vtkMRMLNode *anode)
{
  Superclass::Copy(anode);
  vtkMRMLMeshingWorkflowNode *node = (vtkMRMLMeshingWorkflowNode *) anode;

  this->SetConductance(node->Conductance);
  this->SetNumberOfIterations(node->NumberOfIterations);
  this->SetTimeStep(node->TimeStep);
  this->SetInputVolumeRef(node->InputVolumeRef);
  this->SetOutputVolumeRef(node->OutputVolumeRef);
}

//----------------------------------------------------------------------------
void vtkMRMLMeshingWorkflowNode::PrintSelf(ostream& os, vtkIndent indent)
{
  
  vtkMRMLNode::PrintSelf(os,indent);

  os << indent << "Conductance:   " << this->Conductance << "\n";
  os << indent << "NumberOfIterations:   " << this->NumberOfIterations << "\n";
  os << indent << "TimeStep:   " << this->TimeStep << "\n";
  os << indent << "InputVolumeRef:   " << this->InputVolumeRef << "\n";
  os << indent << "OutputVolumeRef:   " << this->OutputVolumeRef << "\n";
}

