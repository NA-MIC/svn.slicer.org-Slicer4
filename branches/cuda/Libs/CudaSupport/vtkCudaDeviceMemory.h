#ifndef VTKCUDADEVICEMEMORY_H_
#define VTKCUDADEVICEMEMORY_H_

#include "vtkCudaMemory.h"

class VTK_CUDASUPPORT_EXPORT vtkCudaDeviceMemory : public vtkCudaMemory
{
    vtkTypeRevisionMacro(vtkCudaDeviceMemory, vtkCudaMemory);
public:
    static  vtkCudaDeviceMemory* New();

    virtual void* AllocateBytes(size_t byte_count);
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

    virtual bool CopyFrom(vtkImageData* data);
    virtual bool CopyTo(vtkImageData* data);
    virtual bool CopyTo(vtkCudaDeviceMemory* other);
    virtual bool CopyTo(vtkCudaLocalMemory* other);
    virtual bool CopyTo(vtkCudaMemoryArray* other);

    virtual void PrintSelf (ostream &os, vtkIndent indent);

protected:
    vtkCudaDeviceMemory();
    virtual ~vtkCudaDeviceMemory();
    vtkCudaDeviceMemory(const vtkCudaDeviceMemory&);
    vtkCudaDeviceMemory& operator=(const vtkCudaDeviceMemory&);

    void* MemPointer;
};

#endif /*VTKCUDADEVICEMEMORY_H_*/
