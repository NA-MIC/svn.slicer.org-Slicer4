/*=auto=========================================================================

Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

See Doc/copyright/copyright.txt
or http://www.slicer.org/copyright/copyright.txt for details.

Program:   3D Slicer
Module:    $RCSfile: vtkMRMLVolumeNode.cxx,v $
Date:      $Date: 2006/03/17 17:01:53 $
Version:   $Revision: 1.14 $

=========================================================================auto=*/

#include <string>
#include <iostream>
#include <sstream>

#include "vtkObjectFactory.h"
#include "vtkCallbackCommand.h"

#include "vtkMRMLGlyphVolumeNode.h"
#include "vtkMRMLGlyphVolumeSliceDisplayNode.h"
#include "vtkMRMLScene.h"

//------------------------------------------------------------------------------
vtkMRMLGlyphVolumeNode* vtkMRMLGlyphVolumeNode::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLGlyphVolumeNode");
  if(ret)
    {
    return (vtkMRMLGlyphVolumeNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLGlyphVolumeNode;
}

//----------------------------------------------------------------------------

vtkMRMLNode* vtkMRMLGlyphVolumeNode::CreateNodeInstance()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLGlyphVolumeNode");
  if(ret)
    {
    return (vtkMRMLGlyphVolumeNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLGlyphVolumeNode;
}

//----------------------------------------------------------------------------
vtkMRMLGlyphVolumeNode::vtkMRMLGlyphVolumeNode()
{
  this->MaskNodeID = NULL;
}

//----------------------------------------------------------------------------
vtkMRMLGlyphVolumeNode::~vtkMRMLGlyphVolumeNode()
{

  if (this->MaskNodeID)
    {
    delete [] this->MaskNodeID;
    this->MaskNodeID = NULL;
    }

   this->SetAndObserveDisplayNodeID(NULL); 
}

//----------------------------------------------------------------------------
void vtkMRMLGlyphVolumeNode::WriteXML(ostream& of, int nIndent)
{
  Superclass::WriteXML(of, nIndent);
 
  vtkIndent indent(nIndent);
  std::stringstream ss;
  if (this->MaskNodeID != NULL) 
    {
    of << indent << "maskNodeRef=\"" << this->MaskNodeID << "\"";
    }

}

//----------------------------------------------------------------------------
void vtkMRMLGlyphVolumeNode::ReadXMLAttributes(const char** atts)
{

  Superclass::ReadXMLAttributes(atts);

  const char* attName;
  const char* attValue;
  while (*atts != NULL)
    {
    attName = *(atts++);
    attValue = *(atts++);
    std::stringstream ss;
    ss<<attValue;

    if (!strcmp(attName, "maskNodeRef"))
      {
        ss>>this->MaskNodeID;
      }
  }      

} 


//----------------------------------------------------------------------------
// Copy the node's attributes to this object.
// Does NOT copy: ID, FilePrefix, Name, VolumeID
void vtkMRMLGlyphVolumeNode::Copy(vtkMRMLNode *anode)
{
  Superclass::Copy(anode);
  //vtkMRMLGlyphVolumeNode *node = (vtkMRMLGlyphVolumeNode *) anode;

}

//----------------------------------------------------------------------------
vtkMRMLVolumeNode* vtkMRMLGlyphVolumeNode::GetMaskNode()
{
  vtkMRMLVolumeNode* node = NULL;
  if (this->GetScene() && this->GetMaskNodeID() )
    {
    vtkMRMLNode* snode = this->GetScene()->GetNodeByID(this->MaskNodeID);
    node = vtkMRMLVolumeNode::SafeDownCast(snode);
    }
  return node;
}

//----------------------------------------------------------------------------
//vtkMRMLVolumeDisplayNode* vtkMRMLVolumeNode::GetDisplayNode()
//{
//  vtkMRMLGlyphVolumeDisplayNode* node = NULL;
//  if (this->GetScene() && this->GetDisplayNodeID() )
//    {
//    vtkMRMLNode* snode = this->GetScene()->GetNodeByID(this->DisplayNodeID);
//    node = vtkMRMLGlyphVolumeDisplayNode::SafeDownCast(snode);
//    }
//  return node;
//}


//-----------------------------------------------------------
//void vtkMRMLGlyphVolumeNode::UpdateScene(vtkMRMLScene *scene)
//{
//  Superclass::UpdateScene(scene);

//  if (this->GetDiffusionWeightedNodeID()) 
//    {
//    this->SetAndObserveDisplayNodeID(this->GetDiffusionWeightedNodeID());
//    }
//}

//----------------------------------------------------------------------------
void vtkMRMLGlyphVolumeNode::UpdateReferenceID(const char *oldID, const char *newID)
{
  if (this->MaskNodeID && !strcmp(oldID, this->MaskNodeID))
    {
    this->SetMaskNodeID(newID);
    }
  Superclass::UpdateReferenceID(oldID,newID);
}

//-----------------------------------------------------------
void vtkMRMLGlyphVolumeNode::UpdateReferences()
{
  Superclass::UpdateReferences();

if (this->MaskNodeID != NULL && this->Scene->GetNodeByID(this->MaskNodeID) == NULL)
    {
    this->SetMaskNodeID(NULL);
    }
}

//---------------------------------------------------------------------------
void vtkMRMLGlyphVolumeNode::ProcessMRMLEvents ( vtkObject *caller,
                                           unsigned long event, 
                                           void *callData )
{
  Superclass::ProcessMRMLEvents(caller, event, callData);
}

//----------------------------------------------------------------------------
void vtkMRMLGlyphVolumeNode::PrintSelf(ostream& os, vtkIndent indent)
{
  Superclass::PrintSelf(os,indent);
  
  os << indent << "MaskNodeID: " <<
    (this->MaskNodeID ? this->MaskNodeID : "(none)") << "\n";

}

//----------------------------------------------------------------------------
std::vector< vtkMRMLGlyphVolumeSliceDisplayNode*> vtkMRMLGlyphVolumeNode::GetSliceGlyphDisplayNodes()
{
  std::vector< vtkMRMLGlyphVolumeSliceDisplayNode*> nodes;
  int nnodes = this->GetNumberOfDisplayNodes();
  vtkMRMLGlyphVolumeSliceDisplayNode *node = NULL;
  for (int n=0; n<nnodes; n++)
    {
    node = vtkMRMLGlyphVolumeSliceDisplayNode::SafeDownCast(this->GetNthDisplayNode(n));
    if (node) 
      {
      nodes.push_back(node);
      }
    }
  return nodes;
}


 
