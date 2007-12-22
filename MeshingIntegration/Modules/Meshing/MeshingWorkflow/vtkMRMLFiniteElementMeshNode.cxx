/*=auto=========================================================================

Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

See Doc/copyright/copyright.txt
or http://www.slicer.org/copyright/copyright.txt for details.

Program:   3D Slicer
Module:    $RCSfile: vtkMRMLFiniteElementMeshNode.cxx,v $
Date:      $Date: 2006/03/17 15:10:10 $
Version:   $Revision: 1.2 $

=========================================================================auto=*/

#include <string>
#include <iostream>
#include <sstream>

#include "vtkObjectFactory.h"

#include "vtkMRMLFiniteElementMeshNode.h"
#include "vtkMRMLScene.h"


//------------------------------------------------------------------------------
vtkMRMLFiniteElementMeshNode* vtkMRMLFiniteElementMeshNode::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLFiniteElementMeshNode");
  if(ret)
    {
      return (vtkMRMLFiniteElementMeshNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLFiniteElementMeshNode;
}

//----------------------------------------------------------------------------

vtkMRMLFiniteElementMeshNode* vtkMRMLFiniteElementMeshNode::CreateNodeInstance()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLFiniteElementMeshNode");
  if(ret)
    {
      return (vtkMRMLFiniteElementMeshNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLFiniteElementMeshNode;
}

//----------------------------------------------------------------------------
vtkMRMLFiniteElementMeshNode::vtkMRMLFiniteElementMeshNode()
{

   this->DataType = 5;
   this->FileName = NULL;
   this->FilePath = NULL;
}

//----------------------------------------------------------------------------
vtkMRMLFiniteElementMeshNode::~vtkMRMLFiniteElementMeshNode()
{
}

//----------------------------------------------------------------------------
void vtkMRMLFiniteElementMeshNode::WriteXML(ostream& of, int nIndent)
{
Superclass::WriteXML(of, nIndent);

  vtkIndent indent(nIndent);
  {
    std::stringstream ss;
    ss << this->DataType;
    of << indent << "DataType='" << ss.str() << "' ";
  }
  {
    std::stringstream ss;
    ss << this->FileName;
    of << indent << "FileName='" << ss.str() << "' ";
  }
  {
     std::stringstream ss;
     ss << this->FilePath;
     of << indent << "FilePath='" << ss.str() << "' ";
   }
}

//----------------------------------------------------------------------------
void vtkMRMLFiniteElementMeshNode::ReadXMLAttributes(const char** atts)
{

  vtkMRMLNode::ReadXMLAttributes(atts);

  const char* attName;
  const char* attValue;
  while (*atts != NULL) 
    {
    attName = *(atts++);
    attValue = *(atts++);

     if (!strcmp(attName, "DataType")) 
      {
      std::stringstream ss;
      ss << attValue;
      ss >> this->DataType;
      }
    else if (!strcmp(attName, "FileName"))
      {
      std::stringstream ss;
      ss << attValue;
      ss >> this->FileName;
      }
    else if (!strcmp(attName, "FilePath"))
       {
       std::stringstream ss;
       ss << attValue;
       ss >> this->FileName;
       }
    }
}

//----------------------------------------------------------------------------
// Copy the node's attributes to this object.
// Does NOT copy: ID, FilePrefix, Name, VolumeID
void vtkMRMLFiniteElementMeshNode::Copy(vtkMRMLNode *anode)
{
  Superclass::Copy(anode);
  vtkMRMLFiniteElementMeshNode *node = (vtkMRMLFiniteElementMeshNode *) anode;

  this->SetDataType(node->DataType);
  this->SetFileName(node->FileName);
  this->SetFilePath(node->FilePath);
}

//----------------------------------------------------------------------------
void vtkMRMLFiniteElementMeshNode::PrintSelf(ostream& os, vtkIndent indent)
{
  
  vtkMRMLNode::PrintSelf(os,indent);

  os << indent << "DataType:   " << this->DataType << "\n";
  os << indent << "FileName:   " << this->FileName << "\n";
  os << indent << "FilePath:   " << this->FilePath << "\n";

}

