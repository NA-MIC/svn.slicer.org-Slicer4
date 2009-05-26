
/*=========================================================================

  Module:    $RCSfile: vtkFiniteElementMeshList.cxx,v $

  Copyright (c) Kitware, Inc.
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "vtkFiniteElementMeshList.h"
#include "vtkMRMLFiniteElementMeshNode.h"
#include "vtkMRMLScene.h"
#include "vtkDebugLeaks.h"
#include "vtkObject.h"
#include "vtkObjectFactory.h"

#include "vtkMRMLFiniteElementMeshDisplayNode.h"
#include "vtkMRMLFiniteElementMeshOutlineDisplayNode.h"
#include "vtkMRMLUnstructuredGridStorageNode.h"
#include "vtkMimxMeshActor.h"
#include "vtkMRMLColorTableNode.h"

//vtkCxxRevisionMacro(vtkFiniteElementMeshList, "$Revision: 1.3 $");

vtkStandardNewMacro(vtkFiniteElementMeshList);

vtkFiniteElementMeshList::vtkFiniteElementMeshList() 
{ 
    this->SetMRMLSceneForStorage(vtkMRMLScene::GetActiveScene());
    this->actorList = vtkLocalLinkedListWrapper::New();
}

vtkFiniteElementMeshList::~vtkFiniteElementMeshList() 
{
  // need to delete all actors in the list, so there are no dangling references to 
  // the renderwindow.  Delete off of the front of the list
  int NumberOfItemsInList = this->GetNumberOfItems();
  for (int i=0; i<NumberOfItemsInList;i++)
    {
    this->RemoveItem(0);
    vtkDebugMacro("deleting Mesh Actor");
    }
  
}

// save reference to the scene to be used for storage 
void vtkFiniteElementMeshList::SetMRMLSceneForStorage(vtkMRMLScene* scene) 
{
    this->savedMRMLScene = scene;
    // each MRML class type needs to be registeredv
    vtkMRMLFiniteElementMeshNode* meshListNode = vtkMRMLFiniteElementMeshNode::New();
    this->savedMRMLScene->RegisterNodeClass(meshListNode);
    meshListNode->Delete();
    
    vtkMRMLFiniteElementMeshOutlineDisplayNode* meshOutlineNode = vtkMRMLFiniteElementMeshOutlineDisplayNode::New();
    this->savedMRMLScene->RegisterNodeClass(meshOutlineNode);
    meshOutlineNode->Delete();
}


int vtkFiniteElementMeshList::AppendItem(vtkMimxMeshActor* actor)
{
   if (this->savedMRMLScene)
   {
       
     // keep a copy of the actor in the local list.  The geometry in this
     // actor will be shared with the MRML tree.
     this->actorList->AppendItem(actor);
       
     // allocate a new MRML node for this item and add it to the scene
     vtkMRMLFiniteElementMeshNode* newMRMLNode = vtkMRMLFiniteElementMeshNode::New();
     newMRMLNode->SetMimxMeshActor(actor);
     newMRMLNode->SetAndObserveUnstructuredGrid(actor->GetDataSet());
     
     // now add the display and storage nodes
      vtkMRMLFiniteElementMeshDisplayNode* dispNode = vtkMRMLFiniteElementMeshDisplayNode::New();
      vtkMRMLFiniteElementMeshOutlineDisplayNode* dispNode2 = vtkMRMLFiniteElementMeshOutlineDisplayNode::New();
      vtkMRMLUnstructuredGridStorageNode* storeNode = vtkMRMLUnstructuredGridStorageNode::New();
      //vtkMRMLColorTableNode* colorNode = vtkMRMLColorTableNode::New();

      // for this version of the meshing module,we are using the default MRML display of the geometry
      dispNode->SetVisibility(1);
      // point the display node to the proper grid
      dispNode->SetUnstructuredGrid(newMRMLNode->GetUnstructuredGrid());
      dispNode2->SetVisibility(1);
      dispNode2->SetUnstructuredGrid(newMRMLNode->GetUnstructuredGrid());

      // create an object reference from the actor to its corresponding MRML node.
      // this is needed to pass through attribute change calls
      actor->SetMRMLDisplayNode(dispNode);
      actor->SetMRMLOutlineDisplayNode(dispNode2);
   
      // Establish linkage between the surface
      // node and its display and storage nodes, so the viewer will be updated when data
      // or attributes change
      //colorNode->SetScene(this->savedMRMLScene);
      this->savedMRMLScene->AddNodeNoNotify(dispNode);
      this->savedMRMLScene->AddNodeNoNotify(dispNode2);
      this->savedMRMLScene->AddNodeNoNotify(storeNode);
      //this->savedMRMLScene->AddNodeNoNotify(colorNode);
      this->savedMRMLScene->AddNode(newMRMLNode);
      dispNode->SetScene(this->savedMRMLScene);
      dispNode2->SetScene(this->savedMRMLScene);
      storeNode->SetScene(this->savedMRMLScene);
        
      // set the color node to specify the color table associated with the grid
      //dispNode->SetAndObserveColorNodeID(colorNode->GetID());
      // need to turn this on so the scalars are used to color the grid
      //dispNode->SetScalarVisibility(1);
      newMRMLNode->AddAndObserveDisplayNodeID(dispNode->GetID());
      newMRMLNode->AddAndObserveDisplayNodeID(dispNode2->GetID());    
      newMRMLNode->SetAndObserveStorageNodeID(storeNode->GetID());      
     //cout << "copied data to MRML mesh node " << endl;
   } else 
   {
       vtkErrorMacro("MeshingWorkflow: Adding FEMesh to uninitialized MRML Scene");
   }
  return VTK_OK;
}


vtkMimxMeshActor* vtkFiniteElementMeshList::GetItem(vtkIdType id)
{
    return vtkMimxMeshActor::SafeDownCast(this->actorList->GetItem(id));
     
  // first fetch the MRML node that has been requested
//  vtkMRMLFiniteElementMeshNode* requestedMrmlNode = 
//      (vtkMRMLFiniteElementMeshNode*)(this->savedMRMLScene->GetNthNodeByClass(id,"vtkMRMLFiniteElementMeshNode"));
//  return requestedMrmlNode->GetMimxMeshActor();

}


int vtkFiniteElementMeshList::GetNumberOfItems()
{
  return this->actorList->GetNumberOfItems();
  //return this->savedMRMLScene->GetNumberOfNodesByClass("vtkMRMLFiniteElementMeshNode");
}

int vtkFiniteElementMeshList::RemoveItem(int Num)
{
  // keep local list and MRML in sync by deleting both entries
  this->actorList->RemoveItem(Num);
  this->savedMRMLScene->RemoveNode(this->savedMRMLScene->GetNthNodeByClass(Num,"vtkMRMLFiniteElementMeshNode"));
  return VTK_OK;
}
