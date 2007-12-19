#ifndef VTKCUDAMEMORY_H_
#define VTKCUDAMEMORY_H_

#include <stddef.h>

class vtkCudaMemory
{
public:
  static vtkCudaMemory* New();

  virtual void* AllocateBytes(size_t count);
  //BTX
    template<typename T> T* Allocate(size_t count) 
    { return (T*)this->AllocateBytes(count * sizeof(T)); }
  virtual void Free();
  
  virtual void MemSet(int value);

  void* GetMemPointer() const { return this->MemPointer; }
  //BTX
  template<typename T> T* GetMemPointerAs() const { return (T*)this->GetMemPointer(); }
  //ETX

  size_t GetSize() const { return Size; }
  
  ~vtkCudaMemory();
protected:
  vtkCudaMemory();

  void* MemPointer;
  size_t Size;
};



#endif /*VTKCUDAMEMORY_H_*/
