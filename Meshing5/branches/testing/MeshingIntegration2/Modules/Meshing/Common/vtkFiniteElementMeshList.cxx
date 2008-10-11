
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
#include "vtkMRMLUnstructuredGridStorageNode.h"
#include "vtkMimxMeshActor.h"
#include "vtkMRMLColorTableNode.h"

//vtkCxxRevisionMacro(vtkFiniteElementMeshList, "$Revision: 1.3 $");

vtkStandardNewMacro(vtkFiniteElementMeshList);

vtkFiniteElementMeshList::vtkFiniteElementMeshList() 
{ 
    this->SetMRMLSceneForStorage(vtkMRMLScene::GetActiveScene());

}

vtkFiniteElementMeshList::~vtkFiniteElementMeshList() 
{

 
}

// save reference to the scene to be used for storage 
void vtkFiniteElementMeshList::SetMRMLSceneForStorage(vtkMRMLScene* scene) 
{
    this->savedMRMLScene = scene;
    // each MRML class type needs to be registeredv
    vtkMRMLFiniteElementMeshNode* meshListNode = vtkMRMLFiniteElementMeshNode::New();
    this->savedMRMLScene->RegisterNodeClass(meshListNode);
    meshListNode->Delete();
}


int vtkFiniteElementMeshList::AppendItem(vtkMimxMeshActor* actor)
{
   if (this->savedMRMLScene)
   {
     // allocate a new MRML node for this item and add it to the scene
     vtkMRMLFiniteElementMeshNode* newMRMLNode = vtkMRMLFiniteElementMeshNode::New();
     // copy the state variables to the MRML node
     //newMRMLNode->SetFileName(actor->GetFileName());
     //newMRMLNode->SetFilePath(actor->GetFilePath());
//     newMRMLNode->SetFileName("filename");
//     newMRMLNode->SetFilePath("filepath");
//     newMRMLNode->SetDataType(actor->GetDataType());
//     vtkUnstructuredGrid* ugrid = vtkUnstructuredGrid::New();
//     ugrid->DeepCopy(actor->GetDataSet());
//     newMRMLNode->SetAndObserveUnstructuredGrid(ugrid);
     newMRMLNode->SetMimxMeshActor(actor);
     newMRMLNode->SetAndObserveUnstructuredGrid(actor->GetDataSet());
     
     // now add the display and storage nodes
      vtkMRMLFiniteElementMeshDisplayNode* dispNode = vtkMRMLFiniteElementMeshDisplayNode::New();
      vtkMRMLUnstructuredGridStorageNode* storeNode = vtkMRMLUnstructuredGridStorageNode::New();
      vtkMRMLColorTableNode* colorNode = vtkMRMLColorTableNode::New();
      colorNode->SetTypeToRainbow();
       
      // Establish linkage between the surface
      // node and its display and storage nodes, so the viewer will be updated when data
      // or attributes change
      dispNode->SetScene(this->savedMRMLScene);
      storeNode->SetScene(this->savedMRMLScene);
      colorNode->SetScene(this->savedMRMLScene);
      this->savedMRMLScene->AddNodeNoNotify(dispNode);
      this->savedMRMLScene->AddNodeNoNotify(storeNode);
      this->savedMRMLScene->AddNodeNoNotify(colorNode);
      this->savedMRMLScene->AddNode(newMRMLNode);
      // point the display node to the proper grid
      dispNode->SetUnstructuredGrid(newMRMLNode->GetUnstructuredGrid());
      // set the color node to specify the color table associated with the grid
      dispNode->SetAndObserveColorNodeID(colorNode->GetID());
      // need to turn this on so the scalars are used to color the grid
      dispNode->SetScalarVisibility(1);

      newMRMLNode->AddAndObserveDisplayNodeID(dispNode->GetID());
      newMRMLNode->SetAndObserveStorageNodeID(storeNode->GetID());      

     
     cout << "copied data to MRML mesh node " << endl;
   } else 
   {
       vtkErrorMacro("MeshingWorkflow: Adding FEMesh to uninitialized MRML Scene");
   }
  return 0;
}


vtkMimxMeshActor* vtkFiniteElementMeshList::GetItem(vtkIdType id)
{
    //return this->InternalMimxObjectList->GetItem(id);
    vtkMimxMeshActor* returnNode;
     
  // first fetch the MRML node that has been requested
  vtkMRMLFiniteElementMeshNode* requestedMrmlNode = 
      (vtkMRMLFiniteElementMeshNode*)(this->savedMRMLScene->GetNthNodeByClass(id,"vtkMRMLFiniteElementMeshNode"));
          
  // if there is a record in the local list, use the available record, otherwise make
  // a new record to return. 
  
//  if (int nodeCount = this->InternalMimxObjectList->GetNumberOfItems() > 0) {
//      // reuse node from internal list
//      returnNode = this->InternalMimxObjectList->GetItem(nodeCount-1);
//  } else
//  {
//      // allocate a new node
//      vtkMimxUnstructuredGridActor* returnNode = vtkMimxUnstructuredGridActor::New();
//  }
  
  // allocate a new node
  returnNode = vtkMimxMeshActor::New();

  // copy MRML values to the node which we will return to the client
  returnNode->SetFileName(requestedMrmlNode->GetFileName());
  returnNode->SetFilePath(requestedMrmlNode->GetFilePath());
  returnNode->SetDataType(requestedMrmlNode->GetDataType());
  returnNode->GetDataSet()->DeepCopy(requestedMrmlNode->GetUnstructuredGrid());
  return returnNode;
}


int vtkFiniteElementMeshList::GetNumberOfItems()
{
  //return this->InternalMimxObjectList->GetNumberOfItems();
  return this->savedMRMLScene->GetNumberOfNodesByClass("vtkMRMLFiniteElementMeshNode");
}

int vtkFiniteElementMeshList::RemoveItem(int Num)
{

  this->savedMRMLScene->RemoveNode(this->savedMRMLScene->GetNthNodeByClass(Num,"vtkMRMLFiniteElementMeshNode"));
  return 0;
}
