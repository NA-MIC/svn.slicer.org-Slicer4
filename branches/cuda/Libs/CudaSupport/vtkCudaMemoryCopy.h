#ifndef VTKCUDAMEMORYCOPY_H_
#define VTKCUDAMEMORYCOPY_H_

#include "vtkCudaMemoryBase.h"

class vtkCudaMemory;
class vtkCudaHostMemory;
class vtkCudaMemoryArray;
class vtkCudaMemoryPitch;

class VTK_CUDASUPPORTMODULE_EXPORT vtkCudaMemoryCopy
{
public:
  static vtkCudaMemoryCopy* New();

  void SetSource(vtkCudaMemoryBase* source) { this->Source = source; }
  
  vtkCudaMemory* CopyToMemory();
  vtkCudaHostMemory* CopyToHostMemory();
  vtkCudaMemoryPitch* CopyToMemoryPitch();
  vtkCudaMemoryArray* CopyToMemoryArray();

  virtual ~vtkCudaMemoryCopy();
protected:
  vtkCudaMemoryCopy();
  vtkCudaMemoryCopy(const vtkCudaMemoryCopy&);
  vtkCudaMemoryCopy operator=(const vtkCudaMemoryCopy&);
  
  //BTX
  vtkCudaMemoryBase::MemoryType GetSourceType() const;
  //ETX
  vtkCudaMemoryBase* Source;
  vtkCudaMemoryBase* Destination;
};

#endif /*VTKCUDAMEMORYCOPY_H_*/
