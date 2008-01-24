#ifndef VTKCUDAIMAGEDATA_H_
#define VTKCUDAIMAGEDATA_H_

#include "vtkDataSet.h"
#include "vtkVolumeRenderingCudaModule.h"

class vtkCudaMemoryBase;

class VTK_VOLUMERENDERINGCUDAMODULE_EXPORT vtkCudaImageData : public vtkDataSet
{
public:
    vtkTypeRevisionMacro(vtkCudaImageData, vtkDataSet);
    static vtkCudaImageData* New();

    virtual void   PrintSelf (ostream &os, vtkIndent indent);


    virtual void CopyStructure (vtkDataSet *ds);
    virtual vtkIdType   GetNumberOfPoints ();
    virtual vtkIdType   GetNumberOfCells ();
    virtual double *   GetPoint (vtkIdType ptId);
    virtual void   GetPoint (vtkIdType id, double x[3]); /// Not necessary
    virtual vtkCell *   GetCell (vtkIdType cellId);
    virtual void   GetCell (vtkIdType cellId, vtkGenericCell *cell);
    virtual void   GetCellBounds (vtkIdType cellId, double bounds[6]); //// Not necessary
    virtual int   GetCellType (vtkIdType cellId);
    virtual void   GetCellTypes (vtkCellTypes *types);
    virtual void   GetCellPoints (vtkIdType cellId, vtkIdList *ptIds);

    virtual void   Squeeze (); //// Not Necessary;
    virtual void   ComputeBounds (); //// Not Necessary

    virtual void   GetScalarRange (double range[2]); //// Not Necessary

    virtual int   GetMaxCellSize ();
    virtual void   GetPointCells (vtkIdType ptId, vtkIdList *cellIds);
    virtual void   GetCellNeighbors (vtkIdType cellId, vtkIdList *ptIds, vtkIdList *cellIds); //// Not Necessary
    virtual vtkIdType   FindPoint (double x[3]);
    virtual vtkIdType   FindCell (double x[3], vtkCell *cell, vtkIdType cellId, double tol2, int &subId, double pcoords[3], double *weights);
    virtual vtkIdType   FindCell (double x[3], vtkCell *cell, vtkGenericCell *gencell, vtkIdType cellId, double tol2, int &subId, double pcoords[3], double *weights);
    virtual vtkCell *   FindAndGetCell (double x[3], vtkCell *cell, vtkIdType cellId, double tol2, int &subId, double pcoords[3], double *weights); //// Not Necessary



protected:
    vtkCudaImageData();
    virtual ~vtkCudaImageData();

    vtkCudaMemoryBase*   Data;
};

#endif /*VTKCUDAIMAGEDATA_H_*/
