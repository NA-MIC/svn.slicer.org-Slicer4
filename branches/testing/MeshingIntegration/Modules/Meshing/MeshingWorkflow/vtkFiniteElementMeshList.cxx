
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

//vtkCxxRevisionMacro(vtkFiniteElementMeshList, "$Revision: 1.3 $");

vtkStandardNewMacro(vtkFiniteElementMeshList);

vtkFiniteElementMeshList::vtkFiniteElementMeshList() 
{ 
    this->savedMRMLScene = NULL;
    InternalMimxObjectList = vtkLinkedList<vtkMimxUnstructuredGridActor*>::New();
}

vtkFiniteElementMeshList::~vtkFiniteElementMeshList() 
{
    InternalMimxObjectList->Delete();
}

// save reference to the scene to be used for storage 
void vtkFiniteElementMeshList::SetMRMLSceneForStorage(vtkMRMLScene* scene) 
{
    this->savedMRMLScene = scene;
    cout << "*** do we need to register the MRML node classes?" << endl;
    //this->savedMRMLScene->RegisterNodeClass(this);
}


int vtkFiniteElementMeshList::AppendItem(vtkMimxUnstructuredGridActor* actor)
{
   this->InternalMimxObjectList->AppendItem(actor);
  
   if (this->savedMRMLScene)
   {
     // allocate a new MRML node for this item and add it to the scene
     vtkMRMLFiniteElementMeshNode* newMRMLNode = vtkMRMLFiniteElementMeshNode::New();
     // copy the state variables to the MRML node
     newMRMLNode->SetFileName(actor->GetFileName());
     newMRMLNode->SetFilePath(actor->GetFilePath());
     newMRMLNode->SetDataType(actor->GetDataType());
     this->savedMRMLScene->AddNode(newMRMLNode);
     cout << "copied data to MRML bbox node " << endl;
   } else 
   {
       vtkErrorMacro("MeshingWorkflow: Adding to uninitialized MRML Scene");
   }
  return 0;
}


vtkMimxUnstructuredGridActor* vtkFiniteElementMeshList::GetItem(vtkIdType id)
{
    vtkMimxUnstructuredGridActor* returnNode;
    //return this->InternalMimxObjectList->GetItem(id);
    
  // first fetch the MRML node that has been requested
  vtkMRMLFiniteElementMeshNode* requestedMrmlNode = 
      (vtkMRMLFiniteElementMeshNode*)(this->savedMRMLScene->GetNthNodeByClass(id,"vtkMRMLFiniteElementMeshNode"));
          
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
  returnNode->SetFileName(requestedMrmlNode->GetFilePath());
  vtkErrorMacro("need to copy vtkPolyData here");
  return returnNode;
}


int vtkFiniteElementMeshList::GetNumberOfItems()
{
  //return this->InternalMimxObjectList->GetNumberOfItems();
  return this->savedMRMLScene->GetNumberOfNodesByClass("vtkMRMLFiniteElementMeshNode");
}

int vtkFiniteElementMeshList::RemoveItem(int Num)
{
  this->InternalMimxObjectList->RemoveItem(Num);
  this->savedMRMLScene->RemoveNode(this->savedMRMLScene->GetNthNodeByClass(Num,"vtkMRMLFiniteElementMeshNode"));
  return 0;
}
