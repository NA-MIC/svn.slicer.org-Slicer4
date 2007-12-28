#ifndef VTKCUDAIMAGEDATA_H_
#define VTKCUDAIMAGEDATA_H_

#include "vtkDataSet.h"

class vtkCudaMemoryBase;

class vtkCudaImageData : public vtkDataSet
{
public:
  static vtkCudaImageData* New();

protected:
  vtkCudaImageData();
  virtual ~vtkCudaImageData();
  
  vtkCudaMemoryBase*   data;
};

#endif /*VTKCUDAIMAGEDATA_H_*/
