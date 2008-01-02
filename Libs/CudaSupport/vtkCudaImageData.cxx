#include "vtkCudaImageData.h"

#include "vtkCudaMemoryBase.h"

vtkCxxRevisionMacro(vtkCudaImageData, "$ Revision: 1.0 $");

vtkCudaImageData* vtkCudaImageData::New()
{
  return new vtkCudaImageData();
}


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
}

vtkIdType vtkCudaImageData::GetNumberOfCells ()
{
  return 0;
}

double* vtkCudaImageData::GetPoint (vtkIdType ptId)
{
}

void vtkCudaImageData::GetPoint (vtkIdType id, double x[3])
{
}
 /// Not necessary
vtkCell* vtkCudaImageData::GetCell (vtkIdType cellId)
{
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
}

vtkIdType vtkCudaImageData::FindCell (double x[3], vtkCell *cell, vtkIdType cellId, double tol2, int &subId, double pcoords[3], double *weights)
{
}

vtkIdType vtkCudaImageData::FindCell (double x[3], vtkCell *cell, vtkGenericCell *gencell, vtkIdType cellId, double tol2, int &subId, double pcoords[3], double *weights)
{
}

//// Not Necessary
vtkCell* vtkCudaImageData::FindAndGetCell (double x[3], vtkCell *cell, vtkIdType cellId, double tol2, int &subId, double pcoords[3], double *weights)
{
}

