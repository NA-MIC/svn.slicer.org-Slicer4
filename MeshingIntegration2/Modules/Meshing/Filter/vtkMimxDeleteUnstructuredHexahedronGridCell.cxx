/*=========================================================================

Program:   MIMX Meshing Toolkit
Module:    $RCSfile: vtkMimxDeleteUnstructuredHexahedronGridCell.cxx,v $
Language:  C++
Date:      $Date: 2008/03/16 18:33:54 $
Version:   $Revision: 1.6 $

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

#include "vtkMimxDeleteUnstructuredHexahedronGridCell.h"

#include "vtkCell.h"
#include "vtkIdList.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkObjectFactory.h"
#include "vtkPointLocator.h"
#include "vtkPoints.h"
#include "vtkTriangle.h"
#include "vtkUnstructuredGrid.h"
#include "vtkPointData.h"
#include "vtkIntArray.h"
#include "vtkPointLocator.h"
#include "vtkIntArray.h"
#include "vtkCellData.h"


vtkCxxRevisionMacro(vtkMimxDeleteUnstructuredHexahedronGridCell, "$Revision: 1.6 $");
vtkStandardNewMacro(vtkMimxDeleteUnstructuredHexahedronGridCell);

// Description:

vtkMimxDeleteUnstructuredHexahedronGridCell::vtkMimxDeleteUnstructuredHexahedronGridCell()
{
        this->CellNum = -1;
}

vtkMimxDeleteUnstructuredHexahedronGridCell::~vtkMimxDeleteUnstructuredHexahedronGridCell()
{
}

int vtkMimxDeleteUnstructuredHexahedronGridCell::RequestData(
  vtkInformation *vtkNotUsed(request),
  vtkInformationVector **inputVector,
  vtkInformationVector *outputVector)
{
        
  // get the info objects
  vtkInformation *inInfo = inputVector[0]->GetInformationObject(0);
  vtkInformation *outInfo = outputVector->GetInformationObject(0);

  // get the input and output
  vtkUnstructuredGrid *input = vtkUnstructuredGrid::SafeDownCast(
    inInfo->Get(vtkDataObject::DATA_OBJECT()));
  vtkUnstructuredGrid *output = vtkUnstructuredGrid::SafeDownCast(
    outInfo->Get(vtkDataObject::DATA_OBJECT()));
 
  if(this->CellNum < 0 || this->CellNum >= input->GetNumberOfCells())
  {
          vtkErrorMacro("Cell number to be deleted incorrect");
          return 0;
  }

  vtkDebugMacro(<< "Check if the face points input are valid ?");
  vtkIdType numCells = input->GetNumberOfCells();
  vtkIdType numPts = input->GetNumberOfPoints();
  if(numCells < 1 || numPts <8)
  {
          vtkErrorMacro("Invalid input data");
          return 0;
  }
  
  // storage of output
  output->Allocate(input->GetNumberOfCells()-1,input->GetNumberOfCells()-1);
  vtkPointLocator *locator = vtkPointLocator::New();
  vtkPoints *points = vtkPoints::New();
  points->Allocate(input->GetNumberOfPoints());
  locator->InitPointInsertion (points, input->GetPoints()->GetBounds());
  vtkIdList *idlist;
  int i;
  double x[3];
  vtkIdType tempid;
  vtkIntArray *MeshSeedValues = vtkIntArray::SafeDownCast(
                input->GetCellData()->GetArray("Mesh_Seed"));
        vtkIntArray *NewMeshSeedValues = vtkIntArray::New();
        NewMeshSeedValues->SetNumberOfComponents(3);
        //NewMeshSeedValues->SetNumberOfTuples(input->GetNumberOfCells()-1);
        NewMeshSeedValues->SetName("Mesh_Seed");
                        
  for(i=0; i <input->GetNumberOfCells(); i++)
  {
          if(i != this->CellNum)
          {
                  idlist = vtkIdList::New();
                  idlist->DeepCopy(input->GetCell(i)->GetPointIds());
                  for(int j=0; j <8; j++)
                  {
                          input->GetPoints()->GetPoint(idlist->GetId(j),x);              
                          locator->InsertUniquePoint(x,tempid);
                          idlist->SetId(j,tempid);
                  }
                  output->InsertNextCell(12, idlist);
                  idlist->Delete();
                  if ( MeshSeedValues )
                  {
                    int dim[3];
                    MeshSeedValues->GetTupleValue(i,dim);
                    NewMeshSeedValues->InsertNextTupleValue(dim);
                  }
          }
  } 
  points->Squeeze();
  output->SetPoints(points);
  output->Squeeze();
  if ( MeshSeedValues )
  {
    output->GetCellData()->AddArray(NewMeshSeedValues);
  }
  this->CopyConstraintValues(input, output);
  points->Delete();
  locator->Delete();
  NewMeshSeedValues->Delete(); 
 return 1;
  
}
//----------------------------------------------------------------------------------
void vtkMimxDeleteUnstructuredHexahedronGridCell::CopyConstraintValues(
        vtkUnstructuredGrid *input, vtkUnstructuredGrid *output)
{
        vtkIntArray *constrain = vtkIntArray::SafeDownCast(
                input->GetPointData()->GetArray("Constrain"));
        if(constrain)
        {
                vtkPoints *inpoints = input->GetPoints();
                vtkPoints *outpoints = output->GetPoints();
                vtkPointLocator *locator = vtkPointLocator::New();

                int i;
                vtkPoints *points = vtkPoints::New();
                locator->InitPointInsertion(points, outpoints->GetBounds());
                for (i=0; i<inpoints->GetNumberOfPoints(); i++)
                {
                        locator->InsertNextPoint(inpoints->GetPoint(i));
                }

                vtkIntArray *constrainout = vtkIntArray::New();
                constrainout->SetNumberOfValues(outpoints->GetNumberOfPoints());
                for (i=0; i<outpoints->GetNumberOfPoints(); i++)
                {
                        vtkIdType pos;
                        pos = locator->IsInsertedPoint(outpoints->GetPoint(i));
                        if (pos != -1)
                        {
                                constrainout->SetValue(i, constrain->GetValue(pos));
                        }
                        else{
                                constrainout->SetValue(i, 0);
                        }
                }
                constrainout->SetName("Constrain");
                output->GetPointData()->AddArray(constrainout);
                constrainout->Delete();
                locator->Delete();
                points->Delete();
        }
}
//------------------------------------------------------------------------------------

void vtkMimxDeleteUnstructuredHexahedronGridCell::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
