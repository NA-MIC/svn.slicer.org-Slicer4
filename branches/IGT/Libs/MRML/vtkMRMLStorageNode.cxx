/*=auto=========================================================================

Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

See Doc/copyright/copyright.txt
or http://www.slicer.org/copyright/copyright.txt for details.

Program:   3D Slicer
Module:    $RCSfile: vtkMRMLStorageNode.cxx,v $
Date:      $Date: 2007/01/17 20:09:05 $
Version:   $Revision: 1.1.1.1 $

=========================================================================auto=*/

#include <string>
#include <iostream>
#include <sstream>

#include "vtkObjectFactory.h"
#include "vtkCommand.h"
#include "vtkMRMLStorageNode.h"
#include "vtkMRMLScene.h"


//----------------------------------------------------------------------------
vtkMRMLStorageNode::vtkMRMLStorageNode()
{
  this->FileName = NULL;
}

//----------------------------------------------------------------------------
vtkMRMLStorageNode::~vtkMRMLStorageNode()
{
  if (this->FileName) 
    {
    delete [] this->FileName;
    this->FileName = NULL;
    }
}

void vtkMRMLStorageNode::WriteXML(ostream& of, int nIndent)
{
  Superclass::WriteXML(of, nIndent);
  vtkIndent indent(nIndent);

  if (this->FileName != NULL) 
    {
    of << indent << "fileName=\"" << vtkMRMLNode::URLEncodeString(this->FileName) << "\" ";
    }
}

//----------------------------------------------------------------------------
void vtkMRMLStorageNode::ReadXMLAttributes(const char** atts)
{
  Superclass::ReadXMLAttributes(atts);
  const char* attName;
  const char* attValue;
  while (*atts != NULL) 
    {
    attName = *(atts++);
    attValue = *(atts++);
    if (!strcmp(attName, "fileName")) 
      {
      this->SetFileName(vtkMRMLNode::URLDecodeString(attValue));
      }
    }
}

//----------------------------------------------------------------------------
// Copy the node's attributes to this object.
// Does NOT copy: ID, FilePrefix, Name, StorageID
void vtkMRMLStorageNode::Copy(vtkMRMLNode *anode)
{
  Superclass::Copy(anode);
  vtkMRMLStorageNode *node = (vtkMRMLStorageNode *) anode;
  this->SetFileName(node->FileName);

}

//----------------------------------------------------------------------------
void vtkMRMLStorageNode::PrintSelf(ostream& os, vtkIndent indent)
{
  Superclass::PrintSelf(os,indent);
  os << indent << "FileName: " <<
    (this->FileName ? this->FileName : "(none)") << "\n";
}

void vtkMRMLStorageNode::ProcessMRMLEvents ( vtkObject *caller, unsigned long event, void *callData )
{
  if (event ==  vtkCommand::ProgressEvent) 
    {
    this->InvokeEvent ( vtkCommand::ProgressEvent,callData );
    }
}
