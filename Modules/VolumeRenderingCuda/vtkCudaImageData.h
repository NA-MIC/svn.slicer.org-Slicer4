#ifndef VTKCUDAIMAGEDATA_H_
#define VTKCUDAIMAGEDATA_H_

#include "vtkDataSet.h"

class vtkCudaImageData : public vtkDataSet
{
public:
  static vtkCudaImageData* New();


  virtual ~vtkCudaImageData();
protected:
  vtkCudaImageData();
};

#endif /*VTKCUDAIMAGEDATA_H_*/
