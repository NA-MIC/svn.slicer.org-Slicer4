#ifndef VTKCUDAMEMORYBASE_H_
#define VTKCUDAMEMORYBASE_H_

#include <stddef.h>

class vtkCudaMemory;
class vtkCudaMemoryArray;
class vtkCudaMemoryPitch;
class vtkCudaHostMemory;

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
    
//BTX
  typedef enum {
    Undefined,
    
    Memory,
    HostMemory,
    ArrayMemory,
    PitchMemory,
  } MemoryType;
//ETX
    MemoryType GetType() const { return this->Type; }
    
    virtual vtkCudaMemory* CopyToMemory() const { return NULL; }
    virtual vtkCudaHostMemory* CopyToHostMemory() const { return NULL; }
    virtual vtkCudaMemoryArray* CopyToMemoryArray() const { return NULL; }
    virtual vtkCudaMemoryPitch* CopyToMemoryPitch() const { return NULL; }
  
protected:
  vtkCudaMemoryBase();
  virtual ~vtkCudaMemoryBase();
  vtkCudaMemoryBase(const vtkCudaMemoryBase&);
  vtkCudaMemoryBase& operator=(const vtkCudaMemoryBase&);

  size_t Size;    //!< The size of the Allocated Memory
  MemoryType Type;
};

#endif /*VTKCUDAMEMORYBASE_H_*/
