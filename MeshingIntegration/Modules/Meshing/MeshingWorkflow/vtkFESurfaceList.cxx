
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

#include "vtkDebugLeaks.h"
#include "vtkObject.h"
#include "vtkObjectFactory.h"

//vtkCxxRevisionMacro(vtkFESurfaceList, "$Revision: 1.3 $");

vtkStandardNewMacro(vtkFESurfaceList);

vtkFESurfaceList::vtkFESurfaceList() 
{ 
    InternalMimxObjectList = vtkLinkedList<vtkMimxActorBase*>::New();
}

vtkFESurfaceList::~vtkFESurfaceList() 
{
    InternalMimxObjectList->Delete();
}

int vtkFESurfaceList::AppendItem(vtkMimxSurfacePolyDataActor* actor)
{
  return this->InternalMimxObjectList->AppendItem(actor);
}

vtkMimxActorBase* vtkFESurfaceList::GetItem(vtkIdType id)
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
