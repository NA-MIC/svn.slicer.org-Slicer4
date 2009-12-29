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
        // initialize as a typeless list in the beginning, the first append will set the type
        this->ListDatatype = ACTOR_NONE;
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
    // storage is not polymorphic at this time.  
    if (actor->IsA("vtkMimxSurfacePolyDataActor"))
        this->AppendItem(vtkMimxSurfacePolyDataActor::SafeDownCast(actor));
    else if (actor->IsA("vtkMimxUnstructuredGridActor"))
        this->AppendItem(vtkMimxUnstructuredGridActor::SafeDownCast(actor));
    else if (actor->IsA("vtkMimxMeshActor"))
         this->AppendItem(vtkMimxMeshActor::SafeDownCast(actor));
    else if (actor->IsA("vtkMimxImageActor"))
          this->AppendItem(vtkMimxImageActor::SafeDownCast(actor));
    else 
    {
        vtkWarningMacro("Received LinkedList Append on unknown datatype");
        vtkWarningMacro("actor type is:" << actor->GetDataType());
    }

    return 1;
}


int vtkLinkedListWrapper::AppendItem(vtkMimxImageActor* actor)
{
   // Put entry in the local list and the correct MRML list
 
  switch (this->ListDatatype) {
    case ACTOR_NONE: {
      // this is the first append to the list, so set the type and do the operation
      this->ListDatatype=ACTOR_IMAGE;
      return this->List->AppendItem(actor);
      break;
    }
    case ACTOR_IMAGE: {
      return this->List->AppendItem(actor); 
      break;
    }
    default:
      vtkErrorMacro("Attempt to add a non-image to the MimxImageList");
  }

  return 1;
}



int vtkLinkedListWrapper::AppendItem(vtkMimxSurfacePolyDataActor* actor)
{
  switch (this->ListDatatype) {
  case ACTOR_NONE: {
     // this is the first append to the list, so set the type and do the operation
     this->ListDatatype=ACTOR_POLYDATA_SURFACE;
     return this->MRMLSurfaceList->AppendItem(vtkMimxSurfacePolyDataActor::SafeDownCast(actor));
     break;
  }
  case ACTOR_POLYDATA_SURFACE: {
      return this->MRMLSurfaceList->AppendItem(vtkMimxSurfacePolyDataActor::SafeDownCast(actor));
      break;
  }
  default:
    vtkErrorMacro("Attempt to add a non-surface to the MimxSurfaceList");
   }

  return 1;
}


int vtkLinkedListWrapper::AppendItem(vtkMimxUnstructuredGridActor* actor)
{
  switch (this->ListDatatype) {
  case ACTOR_NONE: {
     // this is the first append to the list, so set the type and do the operation
     this->ListDatatype=ACTOR_BUILDING_BLOCK;
     this->MRMLBBlockList->AppendItem(vtkMimxUnstructuredGridActor::SafeDownCast(actor));
     break;
  }
  case ACTOR_BUILDING_BLOCK: {
    return this->MRMLBBlockList->AppendItem(vtkMimxUnstructuredGridActor::SafeDownCast(actor));
    break;
  }
  default:
    vtkErrorMacro("Attempt to add an object of incorrect type to the MimxBuildingBlockList");
   }

  return 1;
}

int vtkLinkedListWrapper::AppendItem(vtkMimxMeshActor* actor)
{

  switch (this->ListDatatype) {
    case ACTOR_NONE: {
       // this is the first append to the list, so set the type and do the operation
       this->ListDatatype=ACTOR_FE_MESH;
       return this->MRMLMeshList->AppendItem(vtkMimxMeshActor::SafeDownCast(actor));
       break;
    }
    case ACTOR_FE_MESH: {
      return this->MRMLMeshList->AppendItem(vtkMimxMeshActor::SafeDownCast(actor));
      break;
    }
    default:
      vtkErrorMacro("Attempt to add an object of incorrect type to the MimxMeshList");
  }

  return 1;
}

vtkMimxActorBase* vtkLinkedListWrapper::GetItem(vtkIdType id)
{
    // since only one of the sublists is filled with content, we will check which type of actor
    // this particular list instance has been storing and return the appropriate item
    switch (this->ListDatatype)
         {
           case ACTOR_POLYDATA_SURFACE: {return this->MRMLSurfaceList->GetItem(id); break;}
           case ACTOR_BUILDING_BLOCK: {return this->MRMLBBlockList->GetItem(id); break;}
           case ACTOR_FE_MESH: {return this->MRMLMeshList->GetItem(id); break;}
           case ACTOR_IMAGE: {return this->List->GetItem(id); break; }
           default: {vtkErrorMacro("attempted retrieval of uninitialized MimxActor list");
               cout << "tried retrieval from uninitialized list instance" << endl;
               return NULL;
               }
         }
}

int vtkLinkedListWrapper::GetNumberOfItems()
{
   // be sure to check which type of objects are being stored in this particular instance and go to 
   // the proper sublist to find and return the number of items in the sublist. 
    switch (this->ListDatatype)
         {
           case ACTOR_POLYDATA_SURFACE: {return this->MRMLSurfaceList->GetNumberOfItems(); break;}
           case ACTOR_BUILDING_BLOCK: {return this->MRMLBBlockList->GetNumberOfItems(); break;}
           case ACTOR_FE_MESH: {return this->MRMLMeshList->GetNumberOfItems(); break;}
           case ACTOR_IMAGE: {return this->List->GetNumberOfItems(); break; }
           // check before list has had an append shouldn't cause an error or warning, just return empty
           case ACTOR_NONE: {return 0; break;} 
           default: {vtkErrorMacro("attempted request of unsupported Mimx Actor Datatype");
               cout << "tried retrieval for unsupported type" << endl;
               return 0;
               }
         }
    
}

int vtkLinkedListWrapper::RemoveItem(int Num)
{
  // be sure to check which type of objects are being stored in this particular instance and go to 
  // the proper sublist to delete the stored item. 
  switch (this->ListDatatype)
  {
      case ACTOR_POLYDATA_SURFACE: {return this->MRMLSurfaceList->RemoveItem(Num); break;}
      case ACTOR_BUILDING_BLOCK: {return this->MRMLBBlockList->RemoveItem(Num); break;}
      case ACTOR_FE_MESH: {return this->MRMLMeshList->RemoveItem(Num); break;}
      case ACTOR_IMAGE: {return this->List->RemoveItem(Num); break; }
      default: {vtkErrorMacro("attempted request of unsupported Mimx Actor Datatype");
            cout << "tried retrieval for unsupported type" << endl;
            return VTK_ERROR;
            }
      }     
}

// initialize the MRML lists for the scene to use for interaction and storage
void vtkLinkedListWrapper::SetMRMLSceneForStorage(vtkMRMLScene* scene)
{
  this->MRMLSurfaceList->SetMRMLSceneForStorage(scene);
  this->MRMLBBlockList->SetMRMLSceneForStorage(scene);
}


vtkMimxActorBase* vtkLinkedListWrapper::GetItemByName(char* objName)
{
    vtkMimxActorBase* matchingEntry;
    bool found = false;
    char entryname[512];
    int count = this->GetNumberOfItems();
    int i = 0;
     
      while (!found &&  i < count)
        {
          // look through the elements in the surface list and see if any of the names match
          if (this->GetItem(i))
            {
              matchingEntry = vtkMimxActorBase::SafeDownCast(this->GetItem(i)); 
              strcpy(entryname,matchingEntry->GetFileName()); 
              cout << "LinkedList: found entry named '" << entryname << "'" << endl;
 
              // set the successful flag if any entry matches
              //found = found || !(strcmp(surfacename,objName));
              found = found || !strcmp(entryname,objName);
           }
          if (found)
               return matchingEntry;
          i++;
        }
      return false;
}

