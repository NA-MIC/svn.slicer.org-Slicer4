#ifndef VTKCUDAMEMORYPITCH_H_
#define VTKCUDAMEMORYPITCH_H_

#include "vtkCudaMemory.h"

class VTK_CUDASUPPORT_EXPORT vtkCudaMemoryPitch : public vtkCudaMemory
{
    vtkTypeRevisionMacro(vtkCudaMemoryPitch, vtkCudaMemory);
public:
    static vtkCudaMemoryPitch* New();

    void* AllocatePitchBytes(size_t width, size_t height, size_t typeSize);
    virtual void Free();

    virtual void MemSet(int value);

    //BTX
    template<typename T> T* AllocatePitch(size_t width, size_t height)
    { return (T*)this->AllocatePitchBytes(width, height, sizeof(T)); }
    //ETX
    size_t GetPitch() const { return this->Pitch; }
    size_t GetWidth() const { return this->Width; } 
    size_t GetHeight() const { return this->Height; }

    virtual void PrintSelf (ostream &os, vtkIndent indent);
protected:
    vtkCudaMemoryPitch();
    virtual ~vtkCudaMemoryPitch();
    vtkCudaMemoryPitch(const vtkCudaMemoryPitch&);
    vtkCudaMemoryPitch& operator=(const vtkCudaMemoryPitch&);

    size_t Pitch;
    size_t Width;
    size_t Height;
};

#endif /*VTKCUDAMEMORYPITCH_H_*/
