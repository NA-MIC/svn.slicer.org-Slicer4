/*=========================================================================

  Module:    $RCSfile: vtkLinkedListWrapper.cxx,v $

  Copyright (c) Kitware, Inc.
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "vtkLinkedListWrapper.h"

#include "vtkDebugLeaks.h"
#include "vtkObject.h"
#include "vtkObjectFactory.h"

#include "vtkFESurfaceList.h"
#include "vtkFiniteElementBuildingBlockList.h"
#include "vtkFiniteElementMeshList.h"
#include "vtkMimxSurfacePolyDataActor.h"
#include "vtkMimxUnstructuredGridActor.h"
#include "vtkMimxMeshActor.h"
#include "vtkMRMLScene.h"

//vtkCxxRevisionMacro(vtkLinkedListWrapper, "$Revision: 1.3 $");

vtkStandardNewMacro(vtkLinkedListWrapper);

vtkLinkedListWrapper::vtkLinkedListWrapper() 
{ 
        List = vtkLinkedList<vtkMimxActorBase*>::New();
        this->MRMLSurfaceList = vtkFESurfaceList::New();
        this->MRMLBBlockList = vtkFiniteElementBuildingBlockList::New();
        this->MRMLMeshList = vtkFiniteElementMeshList::New();
}

vtkLinkedListWrapper::~vtkLinkedListWrapper() 
{
        this->List->Delete();
        this->MRMLSurfaceList->Delete();
        this->MRMLBBlockList->Delete();
        this->MRMLMeshList->Delete();
}

int vtkLinkedListWrapper::AppendItem(vtkMimxActorBase* actor)
{
    // check the datatype to see which list the actor should be added to.  This is necessary because the MRML-backed
    // storage is not polymorphic at this time.  The application must set the datatype vlue to allow objects to 
    // be stored correctly.  Since both buildingBlocks and FEMesh are stored as unstructured grid, this enumeration
    // is the only way to tell the difference and store them in the correct lists, so slicer can attach the correct
    // display nodes. 
    switch (actor->GetDataType())
    {
      case(ACTOR_POLYDATA_SURFACE): {this->MRMLSurfaceList->AppendItem(vtkMimxSurfacePolyDataActor::SafeDownCast(actor)); break;}
      case(ACTOR_BUILDING_BLOCK): {this->MRMLBBlockList->AppendItem(vtkMimxUnstructuredGridActor::SafeDownCast(actor)); break;}
      case(ACTOR_FE_MESH): {this->MRMLMeshList->AppendItem(vtkMimxMeshActor::SafeDownCast(actor)); break;}
      default: vtkErrorMacro("attempted storage of unsupported Mimx Actor Datatype");
      cout << "tried storage for " << actor->GetDataType() << " type" << endl;
    }
    return this->List->AppendItem(actor);
}


vtkMimxActorBase* vtkLinkedListWrapper::GetItem(vtkIdType id)
{
        return this->List->GetItem(id);
}

int vtkLinkedListWrapper::GetNumberOfItems()
{
        return this->List->GetNumberOfItems();
}

int vtkLinkedListWrapper::RemoveItem(int Num)
{
    vtkMimxActorBase *removedactor = this->List->GetItem(Num);
    switch (removedactor->GetDataType())
     {
       case(ACTOR_POLYDATA_SURFACE): {this->MRMLSurfaceList->RemoveItem(Num); break;}
       case(ACTOR_BUILDING_BLOCK): {this->MRMLBBlockList->RemoveItem(Num); break;}
       case(ACTOR_FE_MESH): {this->MRMLMeshList->RemoveItem(Num); break;}
       default: vtkErrorMacro("attempted storage of unsupported Mimx Actor Datatype")
     }
     return this->List->RemoveItem(Num);
}

// initialize the MRML lists for the scene to use for interaction and storage
void vtkLinkedListWrapper::SetMRMLSceneForStorage(vtkMRMLScene* scene)
{
  this->MRMLSurfaceList->SetMRMLSceneForStorage(scene);
  this->MRMLBBlockList->SetMRMLSceneForStorage(scene);
}
