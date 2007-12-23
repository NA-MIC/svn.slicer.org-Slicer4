
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
}


int vtkFESurfaceList::AppendItem(vtkMimxSurfacePolyDataActor* actor)
{
   this->InternalMimxObjectList->AppendItem(actor);
  
  // allocate a new MRML node for this item and add it to the scene
   if (this->savedMRMLScene)
   {
     vtkMRMLFESurfaceNode* newMRMLNode = vtkMRMLFESurfaceNode::New();
     // copy the state variables to the MRML node
     newMRMLNode->SetSurfaceFileName(actor->GetFileName());
     newMRMLNode->SetSurfaceFilePath(actor->GetFilePath());
     newMRMLNode->SetSurfaceDataType(actor->GetDataType());
     this->savedMRMLScene->AddNode(newMRMLNode);
     
     cout << "copied data to MRML node " << endl;
   }
  return 0;
}

vtkMimxSurfacePolyDataActor* vtkFESurfaceList::GetItem(vtkIdType id)
{
  return this->InternalMimxObjectList->GetItem(id);
}

int vtkFESurfaceList::GetNumberOfItems()
{
  return this->InternalMimxObjectList->GetNumberOfItems();
}

int vtkFESurfaceList::RemoveItem(int Num)
{
  return this->InternalMimxObjectList->RemoveItem(Num);
}
