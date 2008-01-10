#ifndef VTKCUDAMEMORYBASE_H_
#define VTKCUDAMEMORYBASE_H_


#include "vtkObject.h"
#include "vtkCudaSupportModule.h"
#include <stddef.h>

class vtkCudaMemory;
class vtkCudaMemoryArray;
class vtkCudaMemoryPitch;
class vtkCudaHostMemory;
class vtkCudaLocalMemory;

class VTK_CUDASUPPORT_EXPORT vtkCudaMemoryBase : public vtkObject
{
public:
    vtkTypeRevisionMacro(vtkCudaMemoryBase, vtkObject);

    static vtkCudaMemoryBase* New();

    /** @brief frees the memory (this must be called in each of the derived destructors) */
    //BTX
    virtual void Free() = 0;
    virtual void MemSet(int value) = 0;
    //ETX
    size_t GetSize() const { return Size; }

    virtual bool CopyTo(vtkCudaMemory* other){ return false; }
    virtual bool CopyTo(vtkCudaLocalMemory* other) { return false; }
    virtual bool CopyTo(vtkCudaMemoryArray* other) { return false; }
    virtual bool CopyTo(vtkCudaMemoryPitch* other) { return false; }

    virtual void PrintSelf (ostream &os, vtkIndent indent);

protected:
    vtkCudaMemoryBase();
    virtual ~vtkCudaMemoryBase();
    vtkCudaMemoryBase(const vtkCudaMemoryBase&);
    vtkCudaMemoryBase& operator=(const vtkCudaMemoryBase&);

    size_t Size;    //!< The size of the Allocated Memory
};

#endif /*VTKCUDAMEMORYBASE_H_*/
