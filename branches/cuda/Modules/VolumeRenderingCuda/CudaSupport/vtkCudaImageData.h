#ifndef VTKCUDAIMAGEDATA_H_
#define VTKCUDAIMAGEDATA_H_

#include "vtkDataSet.h"
#include "vtkCudaSupportModule.h"

class vtkCudaMemoryBase;

class VTK_CUDASUPPORTMODULE_EXPORT vtkCudaImageData : public vtkDataSet
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
