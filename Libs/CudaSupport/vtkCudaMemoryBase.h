#ifndef VTKCUDAMEMORYBASE_H_
#define VTKCUDAMEMORYBASE_H_


#include "vtkObject.h"
#include "vtkCudaSupportModule.h"
#include <stddef.h>

class vtkCudaMemory;
 class vtkCudaDeviceMemory;
 class vtkCudaLocalMemory;
  class vtkCudaHostMemory;
class vtkCudaMemoryArray;
class vtkCudaMemoryPitch;

class VTK_CUDASUPPORT_EXPORT vtkCudaMemoryBase : public vtkObject
{
  friend class vtkCudaMemory;
public:
    vtkTypeRevisionMacro(vtkCudaMemoryBase, vtkObject);

    static vtkCudaMemoryBase* New();

    /** @brief frees the memory (this must be called in each of the derived destructors) */
    //BTX
    virtual void Free() = 0;
    virtual void MemSet(int value) = 0;
    //ETX
    size_t GetSize() const { return Size; }

    //BTX
    //! The Location of the memory is either on the Device or in Main Memory (paged or unpaged) 
    typedef enum {
      MemoryOnDevice, //!< The memory is located on the Device
      MemoryOnHost,  //!< The memory is located on the Host side
    } MemoryLocation;
    
    MemoryLocation GetMemoryLocation() const { return this->Location; }

    virtual bool CopyTo(void* dst, size_t byte_count, size_t offset = 0, MemoryLocation dst_loc = MemoryOnHost) = 0;
    virtual bool CopyFrom(void* src, size_t byte_count, size_t offset = 0, MemoryLocation src_loc = MemoryOnHost) = 0;
    //ETX
    
    // This function does a cast of this to the specified type and then a cast of the other to the specified type, so we are sure from what memory to what we are copying.
    virtual bool CopyTo(vtkCudaMemoryBase* other) { return false; /* To give you a sense what this does:  other->CopyFrom(this); */ }
 
    virtual void PrintSelf (ostream &os, vtkIndent indent);

protected:
    vtkCudaMemoryBase();
    virtual ~vtkCudaMemoryBase();
    vtkCudaMemoryBase(const vtkCudaMemoryBase&);
    vtkCudaMemoryBase& operator=(const vtkCudaMemoryBase&);

    size_t Size;    //!< The size of the Allocated Memory
    //BTX
    MemoryLocation Location;
  //ETX
  virtual bool CopyFrom(vtkCudaMemory* mem) { return false; }
  virtual bool CopyFrom(vtkCudaMemoryPitch* mem) { return false; }
  virtual bool CopyFrom(vtkCudaMemoryArray* mem) { return false; }
};

#endif /*VTKCUDAMEMORYBASE_H_*/
