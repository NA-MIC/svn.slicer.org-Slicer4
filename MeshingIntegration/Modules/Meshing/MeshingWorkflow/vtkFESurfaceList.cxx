
/*=========================================================================

  Module:    $RCSfile: vtkFESurfaceList.cxx,v $

  Copyright (c) Kitware, Inc.
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/


#include "vtkFESurfaceList.h"
#include "vtkMRMLFESurfaceNode.h"
#include "vtkMRMLScene.h"
#include "vtkDebugLeaks.h"
#include "vtkObject.h"
#include "vtkObjectFactory.h"

#include "vtkMRMLModelDisplayNode.h"
#include "vtkMRMLModelStorageNode.h"

//vtkCxxRevisionMacro(vtkFESurfaceList, "$Revision: 1.3 $");

vtkStandardNewMacro(vtkFESurfaceList);

vtkFESurfaceList::vtkFESurfaceList() 
{ 
    this->savedMRMLScene = NULL;
    InternalMimxObjectList = vtkLinkedList<vtkMimxSurfacePolyDataActor*>::New();
}

vtkFESurfaceList::~vtkFESurfaceList() 
{
    InternalMimxObjectList->Delete();
}

// save reference to the scene to be used for storage 
void vtkFESurfaceList::SetMRMLSceneForStorage(vtkMRMLScene* scene) 
{
    this->savedMRMLScene = scene;
    // each MRML node class type has to be registered with the scene
    vtkMRMLFESurfaceNode* newMRMLNode = vtkMRMLFESurfaceNode::New();
     this->savedMRMLScene->RegisterNodeClass( newMRMLNode );
     newMRMLNode->Delete();
  
}


int vtkFESurfaceList::AppendItem(vtkMimxSurfacePolyDataActor* actor)
{
   this->InternalMimxObjectList->AppendItem(actor);
  
  // allocate a new MRML node for this item and add it to the scene
   if (this->savedMRMLScene)
   {
     // create a node to contain the geometry 
     vtkMRMLFESurfaceNode* newMRMLNode = vtkMRMLFESurfaceNode::New();
     this->savedMRMLScene->RegisterNodeClass( newMRMLNode );
     
     // copy the state variables to the MRML node
     newMRMLNode->SetFileName(actor->GetFileName());
     newMRMLNode->SetFilePath(actor->GetFilePath());
     newMRMLNode->SetDataType(actor->GetDataType());
     newMRMLNode->SetAndObservePolyData(actor->GetDataSet());
     this->savedMRMLScene->AddNode(newMRMLNode);
     
     // create node to use for display and storage in slicer; use standard model
      // node initially to learn how display nodes work. Use our own 
      // subclasses later, possibly.  
      vtkMRMLModelDisplayNode* dispNode = vtkMRMLModelDisplayNode::New();
      vtkMRMLModelStorageNode* storeNode = vtkMRMLModelStorageNode::New();
       
      // Establish linkage between the surface
      // node and its display and storage nodes, so the viewer will be updated when data
      // or attributes change
      this->savedMRMLScene->AddNode(dispNode);
      this->savedMRMLScene->AddNode(storeNode);
      newMRMLNode->AddAndObserveDisplayNodeID(dispNode->GetID());
      newMRMLNode->SetStorageNodeID(storeNode->GetID());
   
     
     cout << "copied data to MRML node " << endl;
   }
  return 0;
}


vtkMimxSurfacePolyDataActor* vtkFESurfaceList::GetItem(vtkIdType id)
{ 
   //return this->InternalMimxObjectList->GetItem(id);
   vtkMimxSurfacePolyDataActor* returnNode;
       
   // first fetch the MRML node that has been requested
   vtkMRMLFESurfaceNode* requestedMrmlNode = 
       (vtkMRMLFESurfaceNode*)(this->savedMRMLScene->GetNthNodeByClass(id,"vtkMRMLFESurfaceNode"));
           
   // if there is a record in the local list, use the available record, otherwise make
   // a new record to return. 
   
   if (int nodeCount = this->InternalMimxObjectList->GetNumberOfItems() > 0) {
       // reuse node from internal list
       returnNode = this->InternalMimxObjectList->GetItem(nodeCount-1);
   } else
   {
       // allocate a new node
       vtkMimxSurfacePolyDataActor* returnNode = vtkMimxSurfacePolyDataActor::New();
   }
   // copy MRML values to the node which we will return to the client
   returnNode->SetFileName(requestedMrmlNode->GetFileName());
   returnNode->SetFilePath(requestedMrmlNode->GetFilePath());
   returnNode->SetDataType(requestedMrmlNode->GetDataType());
   // *** deep copy might be unnecessary
   returnNode->GetDataSet()->DeepCopy(requestedMrmlNode->GetPolyData());
   return returnNode;

}

int vtkFESurfaceList::GetNumberOfItems()
{
  //return this->InternalMimxObjectList->GetNumberOfItems();
  return this->savedMRMLScene->GetNumberOfNodesByClass("vtkMRMLFESurfaceNode");
}

int vtkFESurfaceList::RemoveItem(int Num)
{
  return this->InternalMimxObjectList->RemoveItem(Num);
  this->savedMRMLScene->RemoveNode(this->savedMRMLScene->GetNthNodeByClass(Num,"vtkMRMLFESurfaceNode"));
  return 0;
}
