
/*=========================================================================

  Module:    $RCSfile: vtkFiniteElementBoundingBoxList.cxx,v $

  Copyright (c) Kitware, Inc.
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "vtkFiniteElementBoundingBoxList.h"
#include "vtkMRMLFiniteElementBoundingBoxNode.h"
#include "vtkMRMLFiniteElementBoundingBoxDisplayNode.h"
#include "vtkMRMLUnstructuredGridStorageNode.h"
#include "vtkMRMLScene.h"
#include "vtkDebugLeaks.h"
#include "vtkObject.h"
#include "vtkObjectFactory.h"
#include "vtkDataSetWriter.h"

//vtkCxxRevisionMacro(vtkFiniteElementBoundingBoxList, "$Revision: 1.3 $");

vtkStandardNewMacro(vtkFiniteElementBoundingBoxList);

vtkFiniteElementBoundingBoxList::vtkFiniteElementBoundingBoxList() 
{ 
    this->savedMRMLScene = NULL;
    InternalMimxObjectList = vtkLinkedList<vtkMimxUnstructuredGridActor*>::New();
}

vtkFiniteElementBoundingBoxList::~vtkFiniteElementBoundingBoxList() 
{
    InternalMimxObjectList->Delete();
}

// save reference to the scene to be used for storage 
void vtkFiniteElementBoundingBoxList::SetMRMLSceneForStorage(vtkMRMLScene* scene) 
{
    this->savedMRMLScene = scene;
    // each node type should be registered once in the MRML scene, so we do it here when the 
    // MRML scene is set, which is called only once per slicer session. 
    vtkMRMLFiniteElementBoundingBoxNode* feBBNode = vtkMRMLFiniteElementBoundingBoxNode::New();
    this->savedMRMLScene->RegisterNodeClass(feBBNode);
    feBBNode->Delete();
}


int vtkFiniteElementBoundingBoxList::AppendItem(vtkMimxUnstructuredGridActor* actor)
{
   this->InternalMimxObjectList->AppendItem(actor);
  
   if (this->savedMRMLScene)
   {
     // allocate a new MRML node for this item and add it to the scene
     vtkMRMLFiniteElementBoundingBoxNode* newMRMLNode = vtkMRMLFiniteElementBoundingBoxNode::New();
     // copy the state variables to the MRML node
     newMRMLNode->SetFileName(actor->GetFileName());
     newMRMLNode->SetFilePath(actor->GetFilePath());
     newMRMLNode->SetDataType(actor->GetDataType());
     vtkUnstructuredGrid* ugrid = vtkUnstructuredGrid::New();
     ugrid->DeepCopy(actor->GetDataSet());
     newMRMLNode->SetAndObserveUnstructuredGrid(ugrid);
     // *** delete the extra pointer to the ugrid, since MRML node has a reference to it
     //ugrid->Delete();
     this->savedMRMLScene->AddNode(newMRMLNode);
     
     // now add the display and storage nodes
     vtkMRMLFiniteElementBoundingBoxDisplayNode* dispNode = vtkMRMLFiniteElementBoundingBoxDisplayNode::New();
     vtkMRMLUnstructuredGridStorageNode* storeNode = vtkMRMLUnstructuredGridStorageNode::New();
      
     // Establish linkage between the surface
     // node and its display and storage nodes, so the viewer will be updated when data
     // or attributes change
     this->savedMRMLScene->AddNode(dispNode);
     this->savedMRMLScene->AddNode(storeNode);
     newMRMLNode->AddAndObserveDisplayNodeID(dispNode->GetID());
     newMRMLNode->SetStorageNodeID(storeNode->GetID());   

     cout << "copied data to MRML bbox node " << endl;
   } else 
   {
       vtkErrorMacro("MeshingWorkflow: Adding to uninitialized MRML Scene");
   }
  return 0;
}

//-------------------------------------------------------------------------
// This method returns TRUE if the instance variables between an actor and 
// the MRML node match.  This test is run to see if this is the correct MRML
// node to correspond to a modify event. 
//-------------------------------------------------------------------------
/***
bool vtkFiniteElementBoundingBoxList::ItemMatchesMRMLNode(vtkMimxUnstructuredGridActor* actor,
                                vtkMRMLFiniteElementBoundingBoxNode* testMRMLNode)
{
    bool nameMatches = !strcmp(actor->GetFileName(),testMRMLNode->GetFileName());
    bool pathMatches = !strcmp(actor->GetFilePath(),testMRMLNode->GetFilePath());
    bool dataTypeMatches = (actor->GetDataType() == testMRMLNode->GetDataType());
    return nameMatches && pathMatches && dataTypeMatches;
}
****/

int vtkFiniteElementBoundingBoxList::ModifyItem(vtkIdType index, vtkMimxUnstructuredGridActor* actor)
{
  
   if (this->savedMRMLScene)
   {
       // first fetch the MRML node that has been requested
        vtkMRMLFiniteElementBoundingBoxNode* requestedMrmlNode = 
            (vtkMRMLFiniteElementBoundingBoxNode*)(this->savedMRMLScene->GetNthNodeByClass(index,"vtkMRMLFiniteElementBoundingBoxNode"));
        
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
     cout << "modified MRML bbox node: " << index << endl;
   } else 
   {
       vtkErrorMacro("MeshingWorkflow: modifying uninitialized MRML Scene");
   }
  return 0;
}


vtkMimxUnstructuredGridActor* vtkFiniteElementBoundingBoxList::GetItem(vtkIdType id)
{
    //return this->InternalMimxObjectList->GetItem(id);
    vtkMimxUnstructuredGridActor* returnNode;
    
  // first fetch the MRML node that has been requested
  vtkMRMLFiniteElementBoundingBoxNode* requestedMrmlNode = 
      (vtkMRMLFiniteElementBoundingBoxNode*)(this->savedMRMLScene->GetNthNodeByClass(id,"vtkMRMLFiniteElementBoundingBoxNode"));
          
  // if there is a record in the local list, use the available record, otherwise make
  // a new record to return. 
  
  if (int nodeCount = this->InternalMimxObjectList->GetNumberOfItems() > 0) {
      // reuse node from internal list
      returnNode = this->InternalMimxObjectList->GetItem(nodeCount-1);
  } else
  {
      // allocate a new node
      vtkMimxUnstructuredGridActor* returnNode = vtkMimxUnstructuredGridActor::New();
  }
  // copy MRML values to the node which we will return to the client
  returnNode->SetFileName(requestedMrmlNode->GetFileName());
  returnNode->SetFilePath(requestedMrmlNode->GetFilePath());
  returnNode->SetDataType(requestedMrmlNode->GetDataType()); 
  vtkUnstructuredGrid *ugrid = requestedMrmlNode->GetUnstructuredGrid();
  vtkUnstructuredGrid *actorGrid = returnNode->GetDataSet();
  actorGrid = ugrid;
  return returnNode;
}


int vtkFiniteElementBoundingBoxList::GetNumberOfItems()
{
  //return this->InternalMimxObjectList->GetNumberOfItems();
  return this->savedMRMLScene->GetNumberOfNodesByClass("vtkMRMLFiniteElementBoundingBoxNode");
}

int vtkFiniteElementBoundingBoxList::RemoveItem(int Num)
{
  this->InternalMimxObjectList->RemoveItem(Num);
  this->savedMRMLScene->RemoveNode(this->savedMRMLScene->GetNthNodeByClass(Num,"vtkMRMLFiniteElementBoundingBoxNode"));
  return 0;
}
