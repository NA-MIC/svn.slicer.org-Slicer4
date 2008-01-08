#include "vtkCudaImageData.h"

#include "vtkObjectFactory.h"

#include "vtkCudaMemoryBase.h"


vtkCxxRevisionMacro(vtkCudaImageData, "$ Revision: 1.0 $");
vtkStandardNewMacro(vtkCudaImageData);


vtkCudaImageData::vtkCudaImageData()
{
  this->Data = NULL;
}

vtkCudaImageData::~vtkCudaImageData()
{
  if (this->Data != NULL)
    this->Data->Delete();
}

void vtkCudaImageData::PrintSelf (ostream &os, vtkIndent indent)
{
    
}



void vtkCudaImageData::CopyStructure (vtkDataSet *ds)
{
  
}

vtkIdType vtkCudaImageData::GetNumberOfPoints ()
{
  return 0;
}

vtkIdType vtkCudaImageData::GetNumberOfCells ()
{
  return 0;
}

double* vtkCudaImageData::GetPoint (vtkIdType ptId)
{
  return NULL;
}

void vtkCudaImageData::GetPoint (vtkIdType id, double x[3])
{
}
 /// Not necessary
vtkCell* vtkCudaImageData::GetCell (vtkIdType cellId)
{
  return 0;
}

void vtkCudaImageData::GetCell (vtkIdType cellId, vtkGenericCell *cell)
{
}

//// Not necessary
void vtkCudaImageData::GetCellBounds (vtkIdType cellId, double bounds[6])
{
}

int vtkCudaImageData::GetCellType (vtkIdType cellId)
{
  return 0;
}

void vtkCudaImageData::GetCellTypes (vtkCellTypes *types)
{
}

void vtkCudaImageData::GetCellPoints (vtkIdType cellId, vtkIdList *ptIds)
{
}

//// Not Necessary
void vtkCudaImageData::Squeeze ()
{
}

 //// Not Necessary
void vtkCudaImageData::ComputeBounds ()
{
}

 //// Not Necessary
void vtkCudaImageData::GetScalarRange (double range[2])
{
}

int vtkCudaImageData::GetMaxCellSize ()
{
  return 0;
}

void vtkCudaImageData::GetPointCells (vtkIdType ptId, vtkIdList *cellIds)
{
}

//// Not Necessary
void vtkCudaImageData::GetCellNeighbors (vtkIdType cellId, vtkIdList *ptIds, vtkIdList *cellIds)
{
}
vtkIdType vtkCudaImageData::FindPoint (double x[3])
{
    return 0;
}

vtkIdType vtkCudaImageData::FindCell (double x[3], vtkCell *cell, vtkIdType cellId, double tol2, int &subId, double pcoords[3], double *weights)
{
   return 0;
}

vtkIdType vtkCudaImageData::FindCell (double x[3], vtkCell *cell, vtkGenericCell *gencell, vtkIdType cellId, double tol2, int &subId, double pcoords[3], double *weights)
{
   return 0;
}

//// Not Necessary
vtkCell* vtkCudaImageData::FindAndGetCell (double x[3], vtkCell *cell, vtkIdType cellId, double tol2, int &subId, double pcoords[3], double *weights)
{
   return NULL;
}
