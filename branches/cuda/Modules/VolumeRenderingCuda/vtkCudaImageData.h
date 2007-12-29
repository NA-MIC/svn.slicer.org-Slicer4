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

protected:
  vtkCudaImageData();
  virtual ~vtkCudaImageData();
  
  vtkCudaMemoryBase*   data;
};

#endif /*VTKCUDAIMAGEDATA_H_*/
