
/*=========================================================================

  Module:    $RCSfile: vtkFiniteElementBuildingBlockList.cxx,v $

  Copyright (c) Kitware, Inc.
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "vtkFiniteElementBuildingBlockList.h"
#include "vtkMRMLFiniteElementBuildingBlockNode.h"
#include "vtkMRMLFiniteElementBuildingBlockDisplayNode.h"
#include "vtkMRMLUnstructuredGridStorageNode.h"
#include "vtkMRMLScene.h"
#include "vtkDebugLeaks.h"
#include "vtkObject.h"
#include "vtkObjectFactory.h"
#include "vtkDataSetWriter.h"

//vtkCxxRevisionMacro(vtkFiniteElementBuildingBlockList, "$Revision: 1.3 $");

vtkStandardNewMacro(vtkFiniteElementBuildingBlockList);

vtkFiniteElementBuildingBlockList::vtkFiniteElementBuildingBlockList()
{
    this->savedMRMLScene = vtkMRMLScene::GetActiveScene();
    // each node type should be registered once in the MRML scene, so we do it here when the
    // MRML scene is set, which is called only once per slicer session.
    vtkMRMLFiniteElementBuildingBlockNode* feBBNode = vtkMRMLFiniteElementBuildingBlockNode::New();
    vtkMRMLFiniteElementBuildingBlockDisplayNode* feBBDispNode = vtkMRMLFiniteElementBuildingBlockDisplayNode::New();
    //vtkMRMLFiniteElementBuildingBlockStorageNode* feBBStoreNode = vtkMRMLFiniteElementBuildingBlockStorageNode::New();
    this->savedMRMLScene->RegisterNodeClass(feBBNode);
    this->savedMRMLScene->RegisterNodeClass(feBBDispNode);
    feBBNode->Delete();
    feBBDispNode->Delete();
}

vtkFiniteElementBuildingBlockList::~vtkFiniteElementBuildingBlockList()
{
    // need to delete all actors in the list, so there are no dangling references to
    // the renderwindow.  Delete off of the front of the list.  Since the lists own
    // delete method is used, both representations (local and MRML) remain in sync.

//    int NumberOfItemsInList = this->GetNumberOfItems();
//    for (int i=0; i<NumberOfItemsInList; i++)
//    {
//        this->RemoveItem(0);
//        vtkDebugMacro("deleting BBox Actor");
//    }
}

// save reference to the scene to be used for storage
void vtkFiniteElementBuildingBlockList::SetMRMLSceneForStorage(vtkMRMLScene* scene)
{
    // the value passed from the module was NULL, so use the Scene class to return it
    //this->savedMRMLScene = scene;

}


int vtkFiniteElementBuildingBlockList::AppendItem(vtkMimxUnstructuredGridActor* actor)
{
   if (this->savedMRMLScene)
   {
     // add actor to the local list.  A local copy of the list and storage in the
     // MRML tree are kept synchronized constantly through consistent operations of
     // addition, modification, or deletion
   //  this->actorList->AppendItem(actor);

     // allocate a new MRML node for this item and add it to the scene
     vtkMRMLFiniteElementBuildingBlockNode* newMRMLNode = vtkMRMLFiniteElementBuildingBlockNode::New();

     // copy the state variables to the MRML node.  The same UnstructuredGrid
     // instance is pointed to by both the actor and the MRML node.
     newMRMLNode->SetMimxUnstructuredGridActor(actor);
     newMRMLNode->SetAndObserveUnstructuredGrid(actor->GetDataSet());

     // now add the display, storage nodes
     vtkMRMLFiniteElementBuildingBlockDisplayNode* dispNode = vtkMRMLFiniteElementBuildingBlockDisplayNode::New();
     vtkMRMLUnstructuredGridStorageNode* storeNode = vtkMRMLUnstructuredGridStorageNode::New();

     // for this version of the meshing module, rendering is performed through
     // MRML display nodes.
     dispNode->SetVisibility(1);

     // create an object reference from the actor to its corresponding MRML node.
     // this is needed to pass through attribute change calls
     actor->SetMRMLDisplayNode(dispNode);

     dispNode->SetScene(this->savedMRMLScene);
     storeNode->SetScene(this->savedMRMLScene);
     this->savedMRMLScene->AddNode(newMRMLNode);
     this->savedMRMLScene->AddNodeNoNotify(dispNode);
     this->savedMRMLScene->AddNodeNoNotify(storeNode);

     // Establish linkage between the bounding box
     // node and its display and storage nodes, so the viewer will be updated when data
     // or attributes change

     dispNode->SetUnstructuredGrid(newMRMLNode->GetUnstructuredGrid());
     newMRMLNode->AddAndObserveDisplayNodeID(dispNode->GetID());
     newMRMLNode->SetAndObserveStorageNodeID(storeNode->GetID());
     //vtkDebugMacro("copied data to MRML bbox node ");
     newMRMLNode->Modified();

   } else
   {
     vtkErrorMacro("MeshingWorkflow: Adding to uninitialized MRML Scene");
     return VTK_ERROR;
   }
  return VTK_OK;
}



vtkMimxUnstructuredGridActor* vtkFiniteElementBuildingBlockList::GetItem(vtkIdType id)
{

 // return vtkMimxUnstructuredGridActor::SafeDownCast(this->actorList->GetItem(id));
  // first fetch the MRML node that has been requested
  vtkMRMLFiniteElementBuildingBlockNode* requestedMrmlNode =
      (vtkMRMLFiniteElementBuildingBlockNode*)(this->savedMRMLScene->GetNthNodeByClass(id,"vtkMRMLFiniteElementBuildingBlockNode"));
  return requestedMrmlNode->GetMimxUnstructuredGridActor();
}


int vtkFiniteElementBuildingBlockList::GetNumberOfItems()
{
  // it is assumed that the local list will generally be faster than traversing the
  // MRML tree, so use local list to count objects
  //this->actorList->GetNumberOfItems();
  return this->savedMRMLScene->GetNumberOfNodesByClass("vtkMRMLFiniteElementBuildingBlockNode");
}

int vtkFiniteElementBuildingBlockList::RemoveItem(int Num)
{
  //this->actorList->RemoveItem(Num);
  this->savedMRMLScene->RemoveNode(this->savedMRMLScene->GetNthNodeByClass(Num,"vtkMRMLFiniteElementBuildingBlockNode"));
  return VTK_OK;
}
