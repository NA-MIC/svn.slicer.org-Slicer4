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

  virtual void   PrintSelf (ostream &os, vtkIndent indent)


protected:
  vtkCudaImageData();
  virtual ~vtkCudaImageData();
  
  vtkCudaMemoryBase*   Data;
};

#endif /*VTKCUDAIMAGEDATA_H_*/
