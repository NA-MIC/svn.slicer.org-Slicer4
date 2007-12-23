
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
#include "vtkMRMLScene.h"
#include "vtkDebugLeaks.h"
#include "vtkObject.h"
#include "vtkObjectFactory.h"

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
    cout << "*** do we need to register the MRML node classes?" << endl;
    //this->savedMRMLScene->RegisterNodeClass(this);
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
     this->savedMRMLScene->AddNode(newMRMLNode);
     cout << "copied data to MRML bbox node " << endl;
   } else 
   {
       vtkErrorMacro("MeshingWorkflow: Adding to uninitialized MRML Scene");
   }
  return 0;
}


vtkMimxUnstructuredGridActor* vtkFiniteElementBoundingBoxList::GetItem(vtkIdType id)
{
    vtkMimxUnstructuredGridActor* returnNode;
    //return this->InternalMimxObjectList->GetItem(id);
    
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
  
  //vtkErrorMacro("need to copy vtkPolyData here");
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
