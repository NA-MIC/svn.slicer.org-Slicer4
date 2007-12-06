/*=========================================================================

  Module:    $RCSfile: vtkLinkedListWrapper.h,v $

  Copyright (c) Kitware, Inc.
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkFiniteElementBoundingBoxList - a  class that maintains a list of 
//  finite element bounding box objects
// .SECTION Description
// vtkFiniteElementBoundingBoxList manages the storage of several Finite Element
// bbox objects.  Storage is provided in the MRML tree contained in Slicer3.  
// This interface is identifical to the API used by the Univ. of Iowa stand-alone
// Finite Element tools to ease integration between the standalone tools and Slicer.
//
// 

#include "vtkCommon.h"
#include "vtkObject.h"
#include "vtkMimxUnstructuredGridActor.h"
#include "vtkLinkedList.h"
#include "vtkLinkedListWrapper.h"
#include "vtkSetGet.h" // For vtkTypeMacro.

#include "vtkMRMLFiniteElementBoundingBoxNode.h"

// pointer to the scene to use for storage
class vtkMRMLScene;

#ifndef __vtkFiniteElementBoundingBoxList_h
#define __vtkFiniteElementBoundingBoxList_h

class VTK_MIMXCOMMON_EXPORT vtkFiniteElementBoundingBoxList : public vtkLinkedListWrapper
{
public:
  static vtkFiniteElementBoundingBoxList *New();
  vtkTypeMacro(vtkFiniteElementBoundingBoxList, vtkLinkedListWrapper);
//BTX
  vtkLinkedList<vtkMimxUnstructuredGridActor*> *InternalMimxObjectList;
//ETX

  // save reference to the scene to be used for storage 
  void SetMRMLSceneForStorage(vtkMRMLScene* scene);
  
  virtual int AppendItem(vtkMimxUnstructuredGridActor*);
  virtual vtkMimxUnstructuredGridActor* GetItem(vtkIdType);
  virtual int GetNumberOfItems();
  virtual int RemoveItem(int );
protected:
    vtkMRMLScene* savedMRMLScene; 
    vtkFiniteElementBoundingBoxList();
  virtual ~vtkFiniteElementBoundingBoxList();
private:
    vtkFiniteElementBoundingBoxList(const vtkFiniteElementBoundingBoxList&); // Not implemented
   void operator=(const vtkFiniteElementBoundingBoxList&); // Not implemented
};
#endif 



