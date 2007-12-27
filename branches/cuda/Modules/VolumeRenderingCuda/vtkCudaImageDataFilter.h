#ifndef VTKCUDAIMAGEDATAFILTER_H_
#define VTKCUDAIMAGEDATAFILTER_H_

#include "vtkAlgorithm.h"

class vtkCudaImageDataFilter : public vtkAlgorithm
{
public:
  static vtkCudaImageDataFilter* New();

  virtual ~vtkCudaImageDataFilter();

protected:
  vtkCudaImageDataFilter();
};

#endif /*VTKCUDAIMAGEDATAFILTER_H_*/
