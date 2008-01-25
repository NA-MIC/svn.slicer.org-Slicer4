#ifndef VTKCUDAIMAGEDATA_H_
#define VTKCUDAIMAGEDATA_H_

#include "vtkImageData.h"
#include "vtkVolumeRenderingCudaModule.h"

class vtkCudaMemory;

class VTK_VOLUMERENDERINGCUDAMODULE_EXPORT vtkCudaImageData : public vtkImageData
{
public:
    vtkTypeRevisionMacro(vtkCudaImageData, vtkDataSet);
    static vtkCudaImageData* New();

    virtual void PrintSelf (ostream &os, vtkIndent indent);


  //  virtual void CopyStructure (vtkDataSet *ds);

protected:
    vtkCudaImageData();
    virtual ~vtkCudaImageData();

    vtkCudaMemory*   Data;
};

#endif /*VTKCUDAIMAGEDATA_H_*/
