#ifndef VTKCUDAMEMORYBASE_H_
#define VTKCUDAMEMORYBASE_H_

#include <stddef.h>

class vtkCudaMemoryBase
{
public:
  static vtkCudaMemoryBase* New();
  
  /** @brief frees the memory (this must be called in each of the derived destructors) */
  //BTX
  virtual void Free() = 0;
    virtual void MemSet(int value) = 0;
    //ETX
    size_t GetSize() const { return Size; }
  
protected:
  vtkCudaMemoryBase();
  virtual ~vtkCudaMemoryBase();
  vtkCudaMemoryBase(const vtkCudaMemoryBase&);
  vtkCudaMemoryBase& operator=(const vtkCudaMemoryBase&);

  size_t Size;    //!< The size of the Allocated Memory
};

#endif /*VTKCUDAMEMORYBASE_H_*/
