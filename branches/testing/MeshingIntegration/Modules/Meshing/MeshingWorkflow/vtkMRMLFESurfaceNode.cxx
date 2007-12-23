/*=auto=========================================================================

Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

See Doc/copyright/copyright.txt
or http://www.slicer.org/copyright/copyright.txt for details.

Program:   3D Slicer
Module:    $RCSfile: vtkMRMLFESurfaceNode.cxx,v $
Date:      $Date: 2006/03/17 15:10:10 $
Version:   $Revision: 1.2 $

=========================================================================auto=*/

#include <string>
#include <iostream>
#include <sstream>

#include "vtkObjectFactory.h"

#include "vtkMRMLFESurfaceNode.h"
#include "vtkMRMLScene.h"


//------------------------------------------------------------------------------
vtkMRMLFESurfaceNode* vtkMRMLFESurfaceNode::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLFESurfaceNode");
  if(ret)
    {
      return (vtkMRMLFESurfaceNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLFESurfaceNode;
  

}

//----------------------------------------------------------------------------

vtkMRMLModelNode* vtkMRMLFESurfaceNode::CreateNodeInstance()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLFESurfaceNode");
  if(ret)
    {
      return (vtkMRMLFESurfaceNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLFESurfaceNode;
}

//----------------------------------------------------------------------------
vtkMRMLFESurfaceNode::vtkMRMLFESurfaceNode()
{

   this->SurfaceDataType = 2;
   this->SurfaceFileName = NULL;
   this->SurfaceFilePath = NULL;
   this->SurfaceFileName = new char[1024];
   this->SurfaceFilePath = new char[1024];
}

//----------------------------------------------------------------------------
vtkMRMLFESurfaceNode::~vtkMRMLFESurfaceNode()
{
}

//----------------------------------------------------------------------------
void vtkMRMLFESurfaceNode::WriteXML(ostream& of, int nIndent)
{
Superclass::WriteXML(of, nIndent);

  vtkIndent indent(nIndent);
  {
    std::stringstream ss;
    ss << this->SurfaceDataType;
    of << indent << "SurfaceDataType='" << ss.str() << "' ";
  }
  {
    std::stringstream ss;
    ss << this->SurfaceFileName;
    of << indent << "SurfaceFileName='" << ss.str() << "' ";
  }
  {
    std::stringstream ss;
    ss << this->SurfaceFilePath;
    of << indent << "SurfaceFilePath='" << ss.str() << "' ";
  }
}

//----------------------------------------------------------------------------
void vtkMRMLFESurfaceNode::ReadXMLAttributes(const char** atts)
{

  vtkMRMLNode::ReadXMLAttributes(atts);

  const char* attName;
  const char* attValue;
  while (*atts != NULL) 
    {
    attName = *(atts++);
    attValue = *(atts++);

     if (!strcmp(attName, "SurfaceDataType")) 
      {
      std::stringstream ss;
      ss << attValue;
      ss >> this->SurfaceDataType;
      }
    else if (!strcmp(attName, "SurfaceFileName"))
      {
      std::stringstream ss;
      ss << attValue;
      ss >> this->SurfaceFileName;
      }
    else if (!strcmp(attName, "SurfaceFilePath"))
      {
      std::stringstream ss;
      ss << attValue;
      ss >> this->SurfaceFilePath;
      }
    }
}

//----------------------------------------------------------------------------
// Copy the node's attributes to this object.
// Does NOT copy: ID, FilePrefix, Name, VolumeID
void vtkMRMLFESurfaceNode::Copy(vtkMRMLNode *anode)
{
  Superclass::Copy(anode);
  vtkMRMLFESurfaceNode *node = (vtkMRMLFESurfaceNode *) anode;

  this->SetSurfaceDataType(node->SurfaceDataType);
  this->SetSurfaceFileName(node->SurfaceFileName);
  this->SetSurfaceFilePath(node->SurfaceFilePath);
}

//----------------------------------------------------------------------------
void vtkMRMLFESurfaceNode::PrintSelf(ostream& os, vtkIndent indent)
{
  
  vtkMRMLNode::PrintSelf(os,indent);

  os << indent << "SurfaceDataType:   " << this->SurfaceDataType << "\n";
  os << indent << "SurfaceFileName:   " << this->SurfaceFileName << "\n";
  os << indent << "SurfaceFilePath:   " << this->SurfaceFilePath << "\n";
}

