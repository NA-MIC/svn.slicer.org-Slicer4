/*=auto=========================================================================

Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

See Doc/copyright/copyright.txt
or http://www.slicer.org/copyright/copyright.txt for details.

Program:   3D Slicer
Module:    $RCSfile: vtkMRMLDisplayableNode.cxx,v $
Date:      $Date: 2006/03/03 22:26:39 $
Version:   $Revision: 1.3 $

=========================================================================auto=*/
#include <string>
#include <iostream>
#include <sstream>

#include "vtkObjectFactory.h"
#include "vtkCallbackCommand.h"

#include "vtkMRMLDisplayableNode.h"
#include "vtkMRMLScene.h"

//----------------------------------------------------------------------------
vtkMRMLDisplayableNode::vtkMRMLDisplayableNode()
{
  this->PolyData = NULL;
  this->StorageNodeID = NULL;
}

//----------------------------------------------------------------------------
vtkMRMLDisplayableNode::~vtkMRMLDisplayableNode()
{
  this->SetAndObserveDisplayNodeID( NULL);

  this->SetAndObservePolyData(NULL);

  if (this->StorageNodeID) 
    {
    delete [] this->StorageNodeID;
    this->StorageNodeID = NULL;
    }
}

//----------------------------------------------------------------------------
void vtkMRMLDisplayableNode::WriteXML(ostream& of, int nIndent)
{
  // Write all attributes not equal to their defaults
  
  Superclass::WriteXML(of, nIndent);

  vtkIndent indent(nIndent);

  std::stringstream ss;
  unsigned int n;
  for (n=0; n < this->DisplayNodeIDs.size(); n++) 
    {
    ss << this->DisplayNodeIDs[n];
    if (n < DisplayNodeIDs.size()-1)
      {
      ss << " ";
      }
    }
  if (this->DisplayNodeIDs.size() > 0) 
    {
    of << indent << " displayNodeRef=\"" << ss.str().c_str() << "\"";
    }


  if (this->StorageNodeID != NULL) 
    {
    of << indent << " storageNodeRef=\"" << this->StorageNodeID << "\"";
    }
}



//----------------------------------------------------------------------------
void vtkMRMLDisplayableNode::ReadXMLAttributes(const char** atts)
{

  Superclass::ReadXMLAttributes(atts);

  const char* attName;
  const char* attValue;
  while (*atts != NULL) 
    {
    attName = *(atts++);
    attValue = *(atts++);
    if (!strcmp(attName, "displayNodeRef")) 
      {
      std::stringstream ss(attValue);
      while (!ss.eof())
        {
        std::string id;
        ss >> id;
        this->AddDisplayNodeID(id.c_str());
        }

      //this->Scene->AddReferencedNodeID(this->DisplayNodeID, this);
      }    
    else if (!strcmp(attName, "storageNodeRef")) 
      {
      this->SetStorageNodeID(attValue);
      //this->Scene->AddReferencedNodeID(this->StorageNodeID, this);
      }
    }  
}

//----------------------------------------------------------------------------
void vtkMRMLDisplayableNode::UpdateReferenceID(const char *oldID, const char *newID)
{ 
  Superclass::UpdateReferenceID(oldID, newID);
  for (unsigned int i=0; i<this->DisplayNodeIDs.size(); i++)
    {
    if ( std::string(oldID) == this->DisplayNodeIDs[i])
      {
      this->SetNthDisplayNodeID(i, newID);
      }
    }
  if (this->StorageNodeID && !strcmp(oldID, this->StorageNodeID))
    {
    this->SetStorageNodeID(newID);
    return;
    }
}
//----------------------------------------------------------------------------
// Copy the node's attributes to this object.
// Does NOT copy: ID, FilePrefix, Name, ID
void vtkMRMLDisplayableNode::Copy(vtkMRMLNode *anode)
{
  Superclass::Copy(anode);
  vtkMRMLDisplayableNode *node = (vtkMRMLDisplayableNode *) anode;
  this->SetAndObserveDisplayNodeID(NULL);
  int ndnodes = node->GetNumberOfDisplayNodes();
  for (int i=0; i<ndnodes; i++)
    {
    this->SetAndObserveNthDisplayNodeID(i, node->DisplayNodeIDs[i].c_str());
    }

  if (node->PolyData)
    {
    this->SetPolyData(node->PolyData);
    }
  this->SetStorageNodeID(node->StorageNodeID);

}

//----------------------------------------------------------------------------
void vtkMRMLDisplayableNode::PrintSelf(ostream& os, vtkIndent indent)
{
  
  Superclass::PrintSelf(os,indent);
  for (unsigned int i=0; i<this->DisplayNodeIDs.size(); i++)
    {
    os << indent << "DisplayNodeIDs[" << i << "]" <<
      this->DisplayNodeIDs[i] << "\n";
    }
  os << "\nPoly Data:\n";
  if (this->PolyData) 
    {
    this->PolyData->PrintSelf(os, indent.GetNextIndent());
    }
  os << indent << "StorageNodeID: " <<
    (this->StorageNodeID ? this->StorageNodeID : "(none)") << "\n";
  
}

//-----------------------------------------------------------
void vtkMRMLDisplayableNode::UpdateScene(vtkMRMLScene *scene)
{
  Superclass::UpdateScene(scene);
   
  if (this->GetStorageNodeID() == NULL) 
    {
    //vtkErrorMacro("No reference StorageNodeID found");
    return;
    }
  
  vtkMRMLNode* mnode = scene->GetNodeByID(this->StorageNodeID);
  if (mnode) 
    {
    vtkMRMLStorageNode *node  = dynamic_cast < vtkMRMLStorageNode *>(mnode);
    if (node->ReadData(this) == 0)
      {
      scene->SetErrorCode(1);
      std::string msg = std::string("Error reading model file ") + std::string(node->GetFileName());
      scene->SetErrorMessage(msg);
      }
    this->SetAndObservePolyData(this->GetPolyData());
    } 

  for (unsigned int i=0; i<this->DisplayNodes.size(); i++)
    {
    this->DisplayNodes[i]->Delete();
    }
  this->DisplayNodes.clear();

  for (unsigned int i=0; i<this->DisplayNodeIDs.size(); i++)
    {
    vtkMRMLDisplayNode *pnode = vtkMRMLDisplayNode::New();
    this->DisplayNodes.push_back(pnode);

    this->SetAndObserveNthDisplayNodeID(i, this->DisplayNodeIDs[i].c_str());
    }
   
}

//-----------------------------------------------------------
void vtkMRMLDisplayableNode::UpdateReferences()
{
   Superclass::UpdateReferences();
  for (unsigned int i=0; i<this->DisplayNodeIDs.size(); i++)
    {
    if (this->Scene->GetNodeByID(this->DisplayNodeIDs[i]) == NULL)
      {
      this->SetAndObserveNthDisplayNodeID(i, NULL);
      }    
    }

  if (this->StorageNodeID != NULL && this->Scene->GetNodeByID(this->StorageNodeID) == NULL)
    {
    this->SetStorageNodeID(NULL);
    }

}


//----------------------------------------------------------------------------
vtkMRMLDisplayNode* vtkMRMLDisplayableNode::GetNthDisplayNode(int n)
{
  vtkMRMLDisplayNode* node = NULL;
  if (this->GetScene() && this->GetNthDisplayNodeID(n) )
    {
    vtkMRMLNode* snode = this->GetScene()->GetNodeByID(this->GetNthDisplayNodeID(n));
    node = vtkMRMLDisplayNode::SafeDownCast(snode);
    }
  return node;
}

//----------------------------------------------------------------------------
void vtkMRMLDisplayableNode::SetDisplayNodeID(const char *displayNodeID)
{
  if (this->DisplayNodeIDs.empty() && displayNodeID == NULL)
    {
    return;
    }
  if (this->DisplayNodeIDs.size() == 1 && displayNodeID != NULL && this->DisplayNodeIDs[0] == std::string(displayNodeID) )
    {
    return;
    }
  this->DisplayNodeIDs.clear();
  if (displayNodeID != NULL)
    {
    this->DisplayNodeIDs.push_back(std::string(displayNodeID));
    }
  if (displayNodeID) 
    { 
    this->Scene->AddReferencedNodeID(displayNodeID, this); 
    } 
}

//----------------------------------------------------------------------------
void vtkMRMLDisplayableNode::SetNthDisplayNodeID(int n, const char *displayNodeID)
{
  if (this->DisplayNodeIDs.empty() && displayNodeID == NULL)
    {
    return;
    }
  if ((int)this->DisplayNodeIDs.size() <= n)
    {
    return;
    }
  if (displayNodeID != NULL && this->DisplayNodeIDs[n] == std::string(displayNodeID) )
    {
    return;
    }
  if (displayNodeID != NULL)
    {
    this->DisplayNodeIDs[n] = std::string(displayNodeID);
    }
  if (displayNodeID) 
    { 
    this->Scene->AddReferencedNodeID(displayNodeID, this); 
    } 
}

//----------------------------------------------------------------------------
void vtkMRMLDisplayableNode::AddDisplayNodeID(const char *displayNodeID)
{
  if (displayNodeID == NULL)
    {
    return;
    }

  this->DisplayNodeIDs.push_back(std::string(displayNodeID));
  this->Scene->AddReferencedNodeID(displayNodeID, this); 
}

//----------------------------------------------------------------------------
void vtkMRMLDisplayableNode::SetAndObserveDisplayNodeID(const char *displayNodeID)
{
  for (unsigned int i=0; i<this->DisplayNodes.size(); i++)
    {
    if (this->DisplayNodes[i])
      {
      vtkSetAndObserveMRMLObjectMacro(this->DisplayNodes[i], NULL);
      }
    }
  this->DisplayNodes.clear();

  this->SetDisplayNodeID(displayNodeID);

  vtkMRMLDisplayNode *dnode = this->GetDisplayNode();
  this->AddAndObseveDisplayNode(dnode);

  this->Modified(); 

}


//----------------------------------------------------------------------------
void vtkMRMLDisplayableNode::SetAndObserveNthDisplayNodeID(int n, const char *displayNodeID)
{
  if (n >= (int)this->DisplayNodes.size())
    {
    this->AddAndObserveDisplayNodeID(displayNodeID);
    return;
    }
  vtkSetAndObserveMRMLObjectMacro(this->DisplayNodes[n], NULL);
    
  this->SetNthDisplayNodeID(n, displayNodeID);

  vtkMRMLDisplayNode *dnode = this->GetNthDisplayNode(n);
  if (dnode) 
    {
    vtkSetAndObserveMRMLObjectMacro(this->DisplayNodes[n], dnode);
    }
  this->Modified(); 

}

//----------------------------------------------------------------------------
void vtkMRMLDisplayableNode::AddAndObserveDisplayNodeID(const char *displayNodeID)
{

  this->AddDisplayNodeID(displayNodeID);

  vtkMRMLDisplayNode *dnode = this->GetDisplayNode();
  this->AddAndObseveDisplayNode(dnode);
  this->Modified(); 
}

//----------------------------------------------------------------------------
void vtkMRMLDisplayableNode::AddAndObseveDisplayNode(vtkMRMLDisplayNode *dnode)
{
  if (dnode) 
    {
    vtkMRMLDisplayNode *pnode = vtkMRMLDisplayNode::New();
    vtkSetAndObserveMRMLObjectMacro(pnode, dnode);
    this->DisplayNodes.push_back(pnode);
    //pnode->Delete();
    }
}
//----------------------------------------------------------------------------
void vtkMRMLDisplayableNode::SetAndObservePolyData(vtkPolyData *polyData)
{
if (this->PolyData != NULL)
    {
    this->PolyData->RemoveObservers ( vtkCommand::ModifiedEvent, this->MRMLCallbackCommand );
    }

  unsigned long mtime1, mtime2;
  mtime1 = this->GetMTime();
  this->SetPolyData(polyData);
  mtime2 = this->GetMTime();

  if (this->PolyData != NULL)
    {
    this->PolyData->AddObserver ( vtkCommand::ModifiedEvent, this->MRMLCallbackCommand );
    }

  if (mtime1 != mtime2)
    {
    this->InvokeEvent( vtkMRMLDisplayableNode::PolyDataModifiedEvent , this);
    }
}

//----------------------------------------------------------------------------
vtkMRMLStorageNode* vtkMRMLDisplayableNode::GetStorageNode()
{
  vtkMRMLStorageNode* node = NULL;
  if (this->GetScene() && this->GetStorageNodeID() )
    {
    vtkMRMLNode* snode = this->GetScene()->GetNodeByID(this->StorageNodeID);
    node = vtkMRMLStorageNode::SafeDownCast(snode);
    }
  return node;
}


//---------------------------------------------------------------------------
void vtkMRMLDisplayableNode::ProcessMRMLEvents ( vtkObject *caller,
                                           unsigned long event, 
                                           void *callData )
{
  Superclass::ProcessMRMLEvents(caller, event, callData);
  for (unsigned int i=0; i<this->DisplayNodes.size(); i++)
    {
    vtkMRMLDisplayNode *dnode = this->GetNthDisplayNode(i);
    if (dnode != NULL && dnode == vtkMRMLDisplayNode::SafeDownCast(caller) &&
      event ==  vtkCommand::ModifiedEvent)
      {
      this->InvokeEvent(vtkMRMLDisplayableNode::DisplayModifiedEvent, NULL);
      }
    }
  if (this->PolyData == vtkPolyData::SafeDownCast(caller) &&
    event ==  vtkCommand::ModifiedEvent)
    {
    this->ModifiedSinceRead = true;
    this->InvokeEvent(vtkMRMLDisplayableNode::PolyDataModifiedEvent, NULL);
    }
  return;
}

