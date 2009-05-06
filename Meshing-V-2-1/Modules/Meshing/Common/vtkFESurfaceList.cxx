
/*=========================================================================

  Module:    $RCSfile: vtkFESurfaceList.cxx,v $

  Copyright (c) Kitware, Inc.
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/




#include "vtkMRMLScene.h"
#include "vtkDebugLeaks.h"
#include "vtkObject.h"
#include "vtkObjectFactory.h"

#include "vtkMRMLModelDisplayNode.h"
#include "vtkMRMLModelStorageNode.h"

#include "vtkMRMLFESurfaceNode.h"
#include "vtkFESurfaceList.h"

//vtkCxxRevisionMacro(vtkFESurfaceList, "$Revision: 1.3 $");

vtkStandardNewMacro(vtkFESurfaceList);

vtkFESurfaceList::vtkFESurfaceList() 
{ 

  this->actorList = vtkLocalLinkedListWrapper::New();
  
  // get a pointer to the current MRML scene to use for adding/removing nodes
  this->savedMRMLScene = vtkMRMLScene::GetActiveScene();
  
  // register the classes to be stored in the MRML tree
  vtkMRMLFESurfaceNode* newMRMLNode = vtkMRMLFESurfaceNode::New();
  this->savedMRMLScene->RegisterNodeClass(newMRMLNode);
  newMRMLNode->Delete();
}

vtkFESurfaceList::~vtkFESurfaceList() 
{
    // need to delete all actors in the list, so there are no dangling references to 
    // the renderwindow.  Delete off of the front of the list
    int NumberOfItemsInList = this->GetNumberOfItems();
    for (int i=0; i<NumberOfItemsInList; i++)
    {
        this->actorList->RemoveItem(0);
        vtkDebugMacro("deleting Surface Actor");
    }
    this->actorList->Delete();
}

// save reference to the scene to be used for storage 
void vtkFESurfaceList::SetMRMLSceneForStorage(vtkMRMLScene* scene) 
{
    this->savedMRMLScene = scene;
}


int vtkFESurfaceList::AppendItem(vtkMimxSurfacePolyDataActor* actor)
{
    
    // add the actor to the local list
    this->actorList->AppendItem(actor);
    
  // allocate a new MRML node for this item and add it to the scene
   if (this->savedMRMLScene)
   {
     // create a node to contain the geometry 
     vtkMRMLFESurfaceNode* newMRMLNode = vtkMRMLFESurfaceNode::New();
     
//     // if this is the first entry, then initialize the MRML scene
//     if (!registered)
//     {
//       this->savedMRMLScene->RegisterNodeClass( newMRMLNode );
//       registered=1;
//     }
     
     // set the MRML node to point to the same poly data as the actor is using
     newMRMLNode->SetAndObservePolyData(actor->GetDataSet());
     
     // create node to use for display and storage in slicer
      vtkMRMLModelDisplayNode* dispNode = vtkMRMLModelDisplayNode::New();
      vtkMRMLModelStorageNode* storeNode = vtkMRMLModelStorageNode::New();
      
      // for this second version of the meshing module, we are using the MRML display of the geometry
      dispNode->SetVisibility(1);
      
      // *** this broke the mrml reload, why?
      //storeNode->SetFileName(actor->GetFileName());
 
      this->savedMRMLScene->AddNode(newMRMLNode);
      
      // Establish linkage between the surface
      // node and its display and storage nodes, so the viewer will be updated when data
      // or attributes change
      this->savedMRMLScene->AddNode(dispNode);
      this->savedMRMLScene->AddNode(storeNode);
      newMRMLNode->AddAndObserveDisplayNodeID(dispNode->GetID());
      newMRMLNode->SetAndObserveStorageNodeID(storeNode->GetID());
   }
   else {
       vtkErrorMacro("Attempted save to MRML, but scene not initialized");
       return VTK_ERROR;
   }
  return VTK_OK;
}


vtkMimxSurfacePolyDataActor* vtkFESurfaceList::GetItem(vtkIdType id)
{ 
   return vtkMimxSurfacePolyDataActor::SafeDownCast(this->actorList->GetItem(id));
       
//   //  fetch the MRML node that has been requested
//   vtkMRMLFESurfaceNode* requestedMrmlNode = 
//       (vtkMRMLFESurfaceNode*)(this->savedMRMLScene->GetNthNodeByClass(id,"vtkMRMLFESurfaceNode"));
//   // then get the actor from the MRML node and return the actor
//   vtkMimxSurfacePolyDataActor* returnNode = requestedMrmlNode->GetMimxSurfacePolyDataActor();
//   return returnNode;

}

int vtkFESurfaceList::GetNumberOfItems()
{
  // there is a parallel structure, but search through the local list is likely to be faster than a MRML tree traversal
  return this->actorList->GetNumberOfItems();
  //return this->savedMRMLScene->GetNumberOfNodesByClass("vtkMRMLFESurfaceNode");
}

int vtkFESurfaceList::RemoveItem(int Num)
{
  // there is a parallel structure between the the local list and the MRML tree, so both items should be deleted
  this->actorList->RemoveItem(Num);
  
  // to remove object from MRML scene, first delete the storage and display nodes, then the node itself
  vtkMRMLFESurfaceNode* requestedMrmlNode = (vtkMRMLFESurfaceNode*)(this->savedMRMLScene->GetNthNodeByClass(Num,"vtkMRMLFESurfaceNode"));
  this->savedMRMLScene->RemoveNode(requestedMrmlNode->GetDisplayNode());
  this->savedMRMLScene->RemoveNode(requestedMrmlNode->GetStorageNode());
  this->savedMRMLScene->RemoveNode(requestedMrmlNode);
  return VTK_OK;
}
