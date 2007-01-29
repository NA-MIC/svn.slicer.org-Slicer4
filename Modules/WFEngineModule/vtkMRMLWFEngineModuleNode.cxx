/*=auto=========================================================================

Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

See Doc/copyright/copyright.txt
or http://www.slicer.org/copyright/copyright.txt for details.

Program:   3D Slicer
Module:    $RCSfile: vtkMRMLWFEngineModuleNode.cxx,v $
Date:      $Date: 2006/03/17 15:10:10 $
Version:   $Revision: 1.2 $

=========================================================================auto=*/

#include <string>
#include <iostream>
#include <sstream>

#include "vtkObjectFactory.h"

#include "vtkMRMLWFEngineModuleNode.h"
#include "vtkMRMLScene.h"


//------------------------------------------------------------------------------
vtkMRMLWFEngineModuleNode* vtkMRMLWFEngineModuleNode::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLWFEngineModuleNode");
  if(ret)
    {
      return (vtkMRMLWFEngineModuleNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLWFEngineModuleNode;
}

//----------------------------------------------------------------------------

vtkMRMLNode* vtkMRMLWFEngineModuleNode::CreateNodeInstance()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLWFEngineModuleNode");
  if(ret)
    {
      return (vtkMRMLWFEngineModuleNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLWFEngineModuleNode;
}

//----------------------------------------------------------------------------
vtkMRMLWFEngineModuleNode::vtkMRMLWFEngineModuleNode()
{
   this->HideFromEditors = true;
}

//----------------------------------------------------------------------------
vtkMRMLWFEngineModuleNode::~vtkMRMLWFEngineModuleNode()
{
}

//----------------------------------------------------------------------------
void vtkMRMLWFEngineModuleNode::WriteXML(ostream& of, int nIndent)
{
  Superclass::WriteXML(of, nIndent);
  // Write all MRML node attributes into output stream
}

//----------------------------------------------------------------------------
void vtkMRMLWFEngineModuleNode::ReadXMLAttributes(const char** atts)
{
  vtkMRMLNode::ReadXMLAttributes(atts);

  // Read all MRML node attributes from two arrays of names and values
  const char* attName;
  const char* attValue;
  while (*atts != NULL) 
    {
    attName = *(atts++);
    attValue = *(atts++);
    }
}

//----------------------------------------------------------------------------
// Copy the node's attributes to this object.
// Does NOT copy: ID, FilePrefix, Name, VolumeID
void vtkMRMLWFEngineModuleNode::Copy(vtkMRMLNode *anode)
{
  Superclass::Copy(anode);
  vtkMRMLWFEngineModuleNode *node = (vtkMRMLWFEngineModuleNode *) anode;
}

//----------------------------------------------------------------------------
void vtkMRMLWFEngineModuleNode::PrintSelf(ostream& os, vtkIndent indent)
{
  vtkMRMLNode::PrintSelf(os,indent);
}

