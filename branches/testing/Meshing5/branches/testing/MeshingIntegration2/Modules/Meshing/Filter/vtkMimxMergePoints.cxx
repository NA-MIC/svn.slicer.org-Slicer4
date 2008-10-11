/*=========================================================================

Program:   MIMX Meshing Toolkit
Module:    $RCSfile: vtkMimxMergePoints.cxx,v $
Language:  C++
Date:      $Date: 2008/02/08 16:56:02 $
Version:   $Revision: 1.1 $

 Musculoskeletal Imaging, Modelling and Experimentation (MIMX)
 Center for Computer Aided Design
 The University of Iowa
 Iowa City, IA 52242
 http://www.ccad.uiowa.edu/mimx/
 
Copyright (c) The University of Iowa. All rights reserved.
See MIMXCopyright.txt or http://www.ccad.uiowa.edu/mimx/Copyright.htm for details.

This software is distributed WITHOUT ANY WARRANTY; without even 
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
PURPOSE.  See the above copyright notices for more information.

=========================================================================*/

#include "vtkMimxMergePoints.h"

#include "vtkCell.h"
#include "vtkCellData.h"
#include "vtkCellTypes.h"
#include "vtkExecutive.h"
#include "vtkIdList.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkIntArray.h"
#include "vtkMergeCells.h"
#include "vtkObjectFactory.h"
#include "vtkPointData.h"
#include "vtkPoints.h"
#include "vtkPointLocator.h"
#include "vtkUnstructuredGrid.h"

#include "vtkMimxMapOriginalCellAndPointIds.h"

vtkCxxRevisionMacro(vtkMimxMergePoints, "$Revision: 1.1 $");
vtkStandardNewMacro(vtkMimxMergePoints);


vtkMimxMergePoints::vtkMimxMergePoints()
{
        this->SetNumberOfInputPorts(2);
        this->SetNumberOfOutputPorts(2);
        this->Tolerance = 0.0;
}

vtkMimxMergePoints::~vtkMimxMergePoints()
{
}

int vtkMimxMergePoints::RequestData(
  vtkInformation *vtkNotUsed(request),
  vtkInformationVector **inputVector,
  vtkInformationVector *outputVector)
{
 
  // get the info objects
  vtkInformation *completeInfo = inputVector[0]->GetInformationObject(0);
  vtkInformation *partialInfo = inputVector[1]->GetInformationObject(0);
  vtkInformation *outInfo = outputVector->GetInformationObject(0);

   // get the complete and partial mesh
  vtkUnstructuredGrid *completeMesh = vtkUnstructuredGrid::SafeDownCast(
          completeInfo->Get(vtkDataObject::DATA_OBJECT()));
  
  vtkUnstructuredGrid *partialMesh = vtkUnstructuredGrid::SafeDownCast(
          partialInfo->Get(vtkDataObject::DATA_OBJECT()));

  vtkIdType partialnumPts,partialnumCells, completenumPts,completenumCells;
  bool completecellstatus = false, completepointstatus = false;

  vtkUnstructuredGrid *completeOut =  vtkUnstructuredGrid::SafeDownCast(
          outInfo->Get(vtkDataObject::DATA_OBJECT()));
  vtkUnstructuredGrid *partialOut =  vtkUnstructuredGrid::
          SafeDownCast(this->GetExecutive()->GetOutputData(1));

  int i,j;

  // check for the input
  if(!partialMesh)
  {
          vtkErrorMacro("Set PartialMesh File, The input should only be unstructured grid");
          return 0;
  }

  if(!completeMesh)
  {
          vtkErrorMacro("Set CompleteMesh File, The input should only be unstructured grid");
          return 0;
  }

  partialnumPts = partialMesh->GetNumberOfPoints();
  partialnumCells = partialMesh->GetNumberOfCells();

  completenumPts = completeMesh->GetNumberOfPoints();
  completenumCells = completeMesh->GetNumberOfCells();

  if (partialnumPts < 1 || partialnumCells < 1 || 
          completenumPts < 1 || completenumCells < 1)
  {
          vtkErrorMacro(<<"No points to merge");
          return 0;
  }
  
  vtkMimxMapOriginalCellAndPointIds *mappointsandcells = vtkMimxMapOriginalCellAndPointIds::New();
  mappointsandcells->SetCompleteMesh(completeMesh);
  mappointsandcells->SetPartialMesh(partialMesh);
  mappointsandcells->Update();
  mappointsandcells->Delete();

  vtkMergeCells* mergecells = vtkMergeCells::New();
  mergecells->SetUnstructuredGrid(completeOut);
  mergecells->SetPointMergeTolerance(this->Tolerance);
  mergecells->SetTotalNumberOfDataSets(1);
  mergecells->SetTotalNumberOfCells(1000);
  mergecells->SetTotalNumberOfPoints(1000);
  mergecells->MergeDataSet(completeMesh);
  mergecells->Finish();
  mergecells->Delete();
  
  // extract only the cells in merged mesh for merged partial mesh
  vtkPoints *points = vtkPoints::New();
  points->DeepCopy(completeOut->GetPoints());

  vtkUnstructuredGrid *UGrid = vtkUnstructuredGrid::New();
  UGrid->Allocate(1000, 1000);

  vtkIntArray *intarray = vtkIntArray::SafeDownCast(
          partialMesh->GetCellData()->GetArray("Original_Cell_Ids"));

  for (i=0; i<partialMesh->GetNumberOfCells(); i++)
  {
          vtkIdList *locallist = vtkIdList::New();
          locallist->DeepCopy(completeOut->GetCell(intarray->GetValue(i))->GetPointIds());
          UGrid->InsertNextCell(completeOut->GetCellType(intarray->GetValue(i)), locallist);
          locallist->Delete();
  }
  UGrid->SetPoints(points);
  UGrid->BuildLinks();
  // separate the points that are not being used

  vtkPoints *pointsused =vtkPoints::New();
  vtkIdList *pointids = vtkIdList::New();
  
  vtkIdList *ugridids = vtkIdList::New();

  for (i=0; i<points->GetNumberOfPoints(); i++)
  {
          ugridids->Initialize();
          UGrid->GetPointCells(i, ugridids);
          if(ugridids->GetNumberOfIds())
          {
                  pointsused->InsertNextPoint(points->GetPoint(i));
                  pointids->InsertNextId(i);
          }
  }
  ugridids->Delete();
  // partial mesh output
  partialOut->Allocate(1000, 1000);
  partialOut->SetPoints(pointsused);

  for (i=0; i<partialMesh->GetNumberOfCells(); i++)
  {
          vtkIdList *locallist = completeOut->GetCell(intarray->GetValue(i))->GetPointIds();
          vtkIdList *partialpointids = vtkIdList::New();
          partialpointids->SetNumberOfIds(locallist->GetNumberOfIds());

          for (j=0; j<locallist->GetNumberOfIds(); j++)
          {
                 partialpointids->SetId(j, pointids->IsId(locallist->GetId(j)));
          }
          partialOut->InsertNextCell(UGrid->GetCellType(i), partialpointids);
          partialpointids->Delete();
  }
  partialOut->Squeeze();
  
  pointids->Delete();
  pointsused->Delete();
  UGrid->Delete();
  points->Delete();

  return 1;
}
//-----------------------------------------------------------------------------------------
void vtkMimxMergePoints::SetCompleteMesh(vtkUnstructuredGrid *CompleteMesh)
{
        this->SetInput(0, CompleteMesh);
}
//-----------------------------------------------------------------------------------------
void vtkMimxMergePoints::SetPartialMesh(vtkUnstructuredGrid *PartialMesh)
{
        this->SetInput(1, PartialMesh);
}
//------------------------------------------------------------------------------------------
vtkUnstructuredGrid* vtkMimxMergePoints::GetCompleteMergedMesh()
{
        return this->GetOutput();
}
//---------------------------------------------------------------------------
vtkUnstructuredGrid* vtkMimxMergePoints::GetPartialMergedMesh()
{
        return vtkUnstructuredGrid::SafeDownCast(this->GetExecutive()->GetOutputData(1));
}
//------------------------------------------------------------------------------------------
void vtkMimxMergePoints::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
