#ifndef VTKCUDALOCALMEMORY_H_
#define VTKCUDALOCALMEMORY_H_

#include "vtkCudaMemory.h"

class VTK_CUDASUPPORT_EXPORT vtkCudaLocalMemory : public vtkCudaMemory
{
    vtkTypeRevisionMacro(vtkCudaLocalMemory, vtkCudaMemory);
public:
    static vtkCudaLocalMemory* New();

    virtual void* AllocateBytes(size_t count);
    virtual void Free();
    virtual void MemSet(int value); 

    void PrintSelf(ostream& os, vtkIndent indent);

protected:
    vtkCudaLocalMemory();
    virtual ~vtkCudaLocalMemory();
    vtkCudaLocalMemory(const vtkCudaLocalMemory&);
    vtkCudaLocalMemory& operator=(const vtkCudaLocalMemory&);
};

#endif /* VTKCUDALOCALMEMORY_H_ */
