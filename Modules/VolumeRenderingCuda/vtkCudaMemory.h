#ifndef VTKCUDAMEMORY_H_
#define VTKCUDAMEMORY_H_

#include "vtkCudaMemoryBase.h"

class vtkCudaMemory : public vtkCudaMemoryBase
{
public:
  static vtkCudaMemory* New();

  virtual void* AllocateBytes(size_t count);
  //BTX
  template<typename T> T* Allocate(size_t count) 
    { return (T*)this->AllocateBytes(count * sizeof(T)); }
  //ETX

  virtual void Free();
  virtual void MemSet(int value);

  void* GetMemPointer() const { return this->MemPointer; }
  //BTX
  template<typename T> T* GetMemPointerAs() const { return (T*)this->GetMemPointer(); }
  //ETX
  
  ~vtkCudaMemory();
protected:
  vtkCudaMemory();
  vtkCudaMemory(const vtkCudaMemory&);
  vtkCudaMemory& operator=(const vtkCudaMemory&);

  void* MemPointer;
};

#endif /*VTKCUDAMEMORY_H_*/
