#ifndef VTKCUDAHOSTMEMORY_H_
#define VTKCUDAHOSTMEMORY_H_

#include "vtkCudaMemory.h"

//! Cuda Host Memory is page-locked host memory that is directly accesible from the cuda device
/**
 * This memory is known to the graphics card and can be accessed very quickly.
 * 
 * @note to much host memory in page locked mode can reduce overall system performance.
 */
class vtkCudaHostMemory : public vtkCudaMemory
{
public:
  static vtkCudaHostMemory* New();

  virtual void* AllocateBytes(size_t count);
  virtual void Free();
  virtual void MemSet(int value); 

/// TODO make protected
  virtual ~vtkCudaHostMemory();

protected:
  vtkCudaHostMemory();
  vtkCudaHostMemory(const vtkCudaHostMemory&);
  vtkCudaHostMemory& operator=(const vtkCudaHostMemory&);
};

#endif /*VTKCUDAHOSTMEMORY_H_*/
