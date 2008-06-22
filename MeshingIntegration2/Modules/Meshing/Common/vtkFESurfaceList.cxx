
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
    this->savedMRMLScene = NULL;

}

vtkFESurfaceList::~vtkFESurfaceList() 
{

}

// save reference to the scene to be used for storage 
void vtkFESurfaceList::SetMRMLSceneForStorage(vtkMRMLScene* scene) 
{
    this->savedMRMLScene = scene;
    // each MRML node class type has to be registered with the scene

    vtkMRMLFESurfaceNode* newMRMLNode = vtkMRMLFESurfaceNode::New();
    // this->savedMRMLScene->RegisterNodeClass( newMRMLNode );
    vtkMRMLScene::GetActiveScene()->RegisterNodeClass(newMRMLNode);
    this->savedMRMLScene = vtkMRMLScene::GetActiveScene();
     newMRMLNode->Delete();

  
}


int vtkFESurfaceList::AppendItem(vtkMimxSurfacePolyDataActor* actor)
{
    
    static int registered=0;
 
  // allocate a new MRML node for this item and add it to the scene
   if (this->savedMRMLScene)
   {
     // create a node to contain the geometry 
     vtkMRMLFESurfaceNode* newMRMLNode = vtkMRMLFESurfaceNode::New();
     
     // if this is the first entry, then initialize the MRML scene
     if (!registered)
     {
       this->savedMRMLScene->RegisterNodeClass( newMRMLNode );
       registered=1;
     }
     // copy the state variables to the MRML node
     newMRMLNode->SetFileName(actor->GetFileName());
     newMRMLNode->SetFilePath(actor->GetFilePath());
     newMRMLNode->SetDataType(actor->GetDataType());
     newMRMLNode->SetAndObservePolyData(actor->GetDataSet());
     
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
      newMRMLNode->SetAndObserveStorageNodeID(storeNode->GetID());
   
      this->savedMRMLScene->AddNode(newMRMLNode);
      
     cout << "copied data to MRML node " << endl;
   }
   else {
       cout << "Attempted save to MRML, but scene not initialized" << endl;
   }
  return 0;
}


vtkMimxSurfacePolyDataActor* vtkFESurfaceList::GetItem(vtkIdType id)
{ 
   //return this->InternalMimxObjectList->GetItem(id);
       
   // first fetch the MRML node that has been requested
   vtkMRMLFESurfaceNode* requestedMrmlNode = 
       (vtkMRMLFESurfaceNode*)(this->savedMRMLScene->GetNthNodeByClass(id,"vtkMRMLFESurfaceNode"));
   

   // allocate a new node to return data
   // *** todo: cache local objects like the original version of the MRML-list
   vtkMimxSurfacePolyDataActor* returnNode = vtkMimxSurfacePolyDataActor::New();
 
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
  //this->InternalMimxObjectList->RemoveItem(Num);
  this->savedMRMLScene->RemoveNode(this->savedMRMLScene->GetNthNodeByClass(Num,"vtkMRMLFESurfaceNode"));
  return 0;
}
