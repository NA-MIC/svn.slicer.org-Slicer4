/*=========================================================================

  Module:    $RCSfile: vtkLinkedListWrapper.h,v $

  Copyright (c) Kitware, Inc.
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkFESurfaceList - a  class that maintains a list of FE Surface objects
// .SECTION DescriptionrCre
// vtkFESurfaceList manages the storage of several Finite Element
// suface objects.  Storage is provided in the MRML tree contained in Slicer3.  
// This interface is identifical to the API used by the Univ. of Iowa stand-alone
// Finite Element tools to ease integration between the standalone tools and Slicer.
//
// 

#include "vtkCommon.h"
#include "vtkObject.h"
#include "vtkMimxSurfacePolyDataActor.h"
#include "vtkLinkedList.h"
#include "vtkLinkedListWrapper.h"
#include "vtkSetGet.h" // For vtkTypeMacro.

#include "vtkMRMLFESurfaceNode.h"


#ifndef __vtkFESurfaceList_h
#define __vtkFESurfaceList_h

class VTK_MIMXCOMMON_EXPORT vtkFESurfaceList : public vtkLinkedListWrapper
{
public:
  static vtkFESurfaceList *New();
  vtkTypeMacro(vtkFESurfaceList, vtkObject);
//BTX
  vtkLinkedList<vtkMimxActorBase*> *InternalMimxObjectList;
//ETX
  int AppendItem(vtkMimxSurfacePolyDataActor*);
  vtkMimxActorBase* GetItem(vtkIdType);
  int GetNumberOfItems();
  int RemoveItem(int );
protected:
    vtkFESurfaceList();
  virtual ~vtkFESurfaceList();
private:
    vtkFESurfaceList(const vtkFESurfaceList&); // Not implemented
  void operator=(const vtkFESurfaceList&); // Not implemented
};
#endif 



