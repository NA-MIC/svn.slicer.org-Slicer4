
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
    this->savedMRMLScene = NULL;
    this->SetMRMLSceneForStorage(NULL);
  
}

vtkFiniteElementBuildingBlockList::~vtkFiniteElementBuildingBlockList() 
{
    
}

// save reference to the scene to be used for storage 
void vtkFiniteElementBuildingBlockList::SetMRMLSceneForStorage(vtkMRMLScene* scene) 
{
    // the value passed from the module was NULL, so use the Scene class to return it
    //this->savedMRMLScene = scene;
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


int vtkFiniteElementBuildingBlockList::AppendItem(vtkMimxUnstructuredGridActor* actor)
{ 
   if (this->savedMRMLScene)
   {
     // allocate a new MRML node for this item and add it to the scene
     vtkMRMLFiniteElementBuildingBlockNode* newMRMLNode = vtkMRMLFiniteElementBuildingBlockNode::New();

     // copy the state variables to the MRML node
     newMRMLNode->SetFileName(actor->GetFileName());
     newMRMLNode->SetFilePath(actor->GetFilePath());
     newMRMLNode->SetDataType(actor->GetDataType());
//     vtkUnstructuredGrid* ugrid = vtkUnstructuredGrid::New();
//     ugrid->DeepCopy(actor->GetDataSet());
//     newMRMLNode->SetAndObserveUnstructuredGrid(ugrid);
     newMRMLNode->SetAndObserveUnstructuredGrid(actor->GetDataSet());
     
     // now add the display, storage, and displayable nodes
     vtkMRMLFiniteElementBuildingBlockDisplayNode* dispNode = vtkMRMLFiniteElementBuildingBlockDisplayNode::New();
     vtkMRMLUnstructuredGridStorageNode* storeNode = vtkMRMLUnstructuredGridStorageNode::New();

     dispNode->SetScene(this->savedMRMLScene);
     storeNode->SetScene(this->savedMRMLScene);
     //storeNode->SetFileName(newMRMLNode->GetFileName());
     this->savedMRMLScene->AddNodeNoNotify(dispNode);
     this->savedMRMLScene->AddNodeNoNotify(storeNode);
     this->savedMRMLScene->AddNode(newMRMLNode);

     // Establish linkage between the bounding box
     // node and its display and storage nodes, so the viewer will be updated when data
     // or attributes change
     dispNode->SetUnstructuredGrid(newMRMLNode->GetUnstructuredGrid());
     newMRMLNode->AddAndObserveDisplayNodeID(dispNode->GetID());
     newMRMLNode->SetAndObserveStorageNodeID(storeNode->GetID());   

     vtkDebugMacro("copied data to MRML bbox node ");
     newMRMLNode->Modified();
     
   } else 
   {
     vtkErrorMacro("MeshingWorkflow: Adding to uninitialized MRML Scene");
   }
  return 0;
}



int vtkFiniteElementBuildingBlockList::ModifyItem(vtkIdType index, vtkMimxUnstructuredGridActor* actor)
{
  
   if (this->savedMRMLScene)
   {
       // first fetch the MRML node that has been requested
        vtkMRMLFiniteElementBuildingBlockNode* requestedMrmlNode = 
            (vtkMRMLFiniteElementBuildingBlockNode*)(this->savedMRMLScene->GetNthNodeByClass(index,"vtkMRMLFiniteElementBuildingBlockNode"));
        
//        vtkDataSetWriter *oldwrite = vtkDataSetWriter::New();
//        oldwrite->SetFileName("oldgrid.vtk");
//        oldwrite->SetInput(requestedMrmlNode->GetUnstructuredGrid());
//        oldwrite->Write();
//        
//        vtkDataSetWriter *modwrite = vtkDataSetWriter::New();
//        modwrite->SetFileName("grid-to-modify-with.vtk");
//        modwrite->SetInput(actor->GetDataSet());
//        modwrite->Write();       
//        
    // copy the state variables to the MRML node
     requestedMrmlNode->SetFileName(actor->GetFileName());
     requestedMrmlNode->SetFilePath(actor->GetFilePath());
     requestedMrmlNode->SetDataType(actor->GetDataType());
     // delete the old ugrid
//     requestedMrmlNode->GetUnstructuredGrid()->Delete();
//     vtkUnstructuredGrid* ugrid = vtkUnstructuredGrid::New();
//     ugrid->DeepCopy(actor->GetDataSet());
//     requestedMrmlNode->SetAndObserveUnstructuredGrid(ugrid);
     requestedMrmlNode->GetUnstructuredGrid()->DeepCopy(actor->GetDataSet());

//     vtkDataSetWriter *newwrite = vtkDataSetWriter::New();
//     newwrite->SetFileName("modified-mrml-grid.vtk");
//     newwrite->SetInput(requestedMrmlNode->GetUnstructuredGrid());
//     newwrite->Write();       

//     requestedMrmlNode->GetUnstructuredGrid()->DeepCopy(actor->GetDataSet());
      
     // *** delete this reference? 
     //ugrid->Delete();
     requestedMrmlNode->Modified();
     
     cout << "modified MRML bbox node: " << index << endl;
   } else 
   {
       vtkErrorMacro("MeshingWorkflow: modifying uninitialized MRML Scene");
   }
  return 0;
}


vtkMimxUnstructuredGridActor* vtkFiniteElementBuildingBlockList::GetItem(vtkIdType id)
{
    //return this->InternalMimxObjectList->GetItem(id);
    vtkMimxUnstructuredGridActor* returnNode;
    
  // first fetch the MRML node that has been requested
  vtkMRMLFiniteElementBuildingBlockNode* requestedMrmlNode = 
      (vtkMRMLFiniteElementBuildingBlockNode*)(this->savedMRMLScene->GetNthNodeByClass(id,"vtkMRMLFiniteElementBuildingBlockNode"));
          
//  // if there is a record in the local list, use the available record, otherwise make
//  // a new record to return. 
//  if (int nodeCount = this->InternalMimxObjectList->GetNumberOfItems() > 0) {
//      // reuse node from internal list
//      returnNode = this->InternalMimxObjectList->GetItem(nodeCount-1);
//  } else
//  {
//      // allocate a new node
//      vtkMimxUnstructuredGridActor* returnNode = vtkMimxUnstructuredGridActor::New();
//  }
  
  // allocate a new node
  returnNode = vtkMimxUnstructuredGridActor::New();

  
  // copy MRML values to the node which we will return to the client
  returnNode->SetFileName(requestedMrmlNode->GetFileName());
  returnNode->SetFilePath(requestedMrmlNode->GetFilePath());
  returnNode->SetDataType(requestedMrmlNode->GetDataType()); 
  vtkUnstructuredGrid *ugrid = requestedMrmlNode->GetUnstructuredGrid();
  vtkUnstructuredGrid *actorGrid = returnNode->GetDataSet();
  actorGrid = ugrid;
  return returnNode;
}


int vtkFiniteElementBuildingBlockList::GetNumberOfItems()
{
  return this->savedMRMLScene->GetNumberOfNodesByClass("vtkMRMLFiniteElementBuildingBlockNode");
}

int vtkFiniteElementBuildingBlockList::RemoveItem(int Num)
{
  this->savedMRMLScene->RemoveNode(this->savedMRMLScene->GetNthNodeByClass(Num,"vtkMRMLFiniteElementBuildingBlockNode"));
  return 0;
}
