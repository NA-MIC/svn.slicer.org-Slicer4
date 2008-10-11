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
        vtkWarningMacro("Received LinkedList Append on unknown datatype");
   

}


int vtkLinkedListWrapper::AppendItem(vtkMimxImageActor* actor)
{
   // Put entry in the local list and the correct MRML list
   //this->MRMLSurfaceList->AppendItem(vtkMimxSurfacePolyDataActor::SafeDownCast(actor));
   return this->List->AppendItem(actor);
}



int vtkLinkedListWrapper::AppendItem(vtkMimxSurfacePolyDataActor* actor)
{
   // Put entry in the local list and the correct MRML list
   this->MRMLSurfaceList->AppendItem(vtkMimxSurfacePolyDataActor::SafeDownCast(actor));
   return this->List->AppendItem(actor);
}

int vtkLinkedListWrapper::AppendItem(vtkMimxUnstructuredGridActor* actor)
{
   // Put entry in the local list and the correct MRML list
   this->MRMLBBlockList->AppendItem(vtkMimxUnstructuredGridActor::SafeDownCast(actor));
   return this->List->AppendItem(actor);
}

int vtkLinkedListWrapper::AppendItem(vtkMimxMeshActor* actor)
{
   // Put entry in the local list and the correct MRML list
   this->MRMLMeshList->AppendItem(vtkMimxMeshActor::SafeDownCast(actor));
   return this->List->AppendItem(actor);
}

vtkMimxActorBase* vtkLinkedListWrapper::GetItem(vtkIdType id)
{
        return this->List->GetItem(id);
//        switch (datatype)
//         {
//           case(ACTOR_POLYDATA_SURFACE): {return this->MRMLSurfaceList->GetItem(id); break;}
//           case(ACTOR_BUILDING_BLOCK): {return this->MRMLBBlockList->GetItem(id); break;}
//           case(ACTOR_FE_MESH): {return this->MRMLMeshList->GetItem(id); break;}
//           default: {vtkErrorMacro("attempted retrieval of unsupported Mimx Actor Datatype");
//               cout << "tried retrieval for " << actor->GetDataType() << " type" << endl;
//               return NULL;
//               }
//         }
}

int vtkLinkedListWrapper::GetNumberOfItems()
{
    return this->List->GetNumberOfItems();
}

int vtkLinkedListWrapper::RemoveItem(int Num)
{
    
}

//*** this fails because there are dummy entries in the mrml tree coming from somewhere.
// debug this after switching to the new code base. 

//int vtkLinkedListWrapper::RemoveItem(int Num)
//{
//    vtkMimxActorBase *removedactor = this->List->GetItem(Num);
//
//    // because the entries in the local list are spread out across the different MRML list types, 
//    // we need to do a search through the mrml lists for a match of the actor.  We can check that
//    // the actors are equal because the actor instances are shared between the local and MRML lists
//    // to save size and eliminate redundancy.  The application level can also add values in any order
//    // to the actors and the MRML tree is guaranteed to get all the correct values. 
//    
//    int found = 0;
//    int index;
//    
//    // look in each list successively but abort as soon as a match is found.
//    // A dummy for loop is used here so we have something to break out of
//    
//    for (int dummy=0;dummy<1;dummy++)
//    {
//        cout << "looking in surface list" << endl;
//        index=0;
//        while ( (index < this->MRMLSurfaceList->GetNumberOfItems()) & !found) 
//        {
//            if (this->MRMLSurfaceList->GetItem(index) == removedactor)
//            {
//              found = 1; 
//              this->MRMLSurfaceList->RemoveItem(index);
//              cout << "found match in surface list; removed it" << endl;
//              break;
//            }
//            index++;
//        }  
//        
//        if (found) break;        // quit if we already found a match  
//        index=0;
//        cout << "looking in bblock list" << endl;
//         while ( (index < this->MRMLBBlockList->GetNumberOfItems()) & !found) 
//         {
//             if (this->MRMLBBlockList->GetItem(index) == removedactor)
//             {
//               found = 1; 
//               this->MRMLBBlockList->RemoveItem(index);
//               cout << "found match in bblock list; removed it" << endl;
//               break;
//             }
//             index++;
//         }  
//
//         if (found) break;        // quit if we already found a match
//         cout << "looking in mesh list" << endl;
//         index=0;
//           while ( (index < this->MRMLMeshList->GetNumberOfItems()) & !found) 
//           {
//               if (this->MRMLMeshList->GetItem(index) == removedactor)
//               {
//                 found = 1; 
//                 this->MRMLMeshList->RemoveItem(index);
//                 cout << "found match in mesh list; removed it" << endl;
//                 break;
//               }
//               index++;
//           } 
//    }
//    // also remove instance from local list
//    // *** I wonder if we have to worry about double deletion of the actor here?
//    return this->List->RemoveItem(Num);
//}

// initialize the MRML lists for the scene to use for interaction and storage
void vtkLinkedListWrapper::SetMRMLSceneForStorage(vtkMRMLScene* scene)
{
  this->MRMLSurfaceList->SetMRMLSceneForStorage(scene);
  this->MRMLBBlockList->SetMRMLSceneForStorage(scene);
}
