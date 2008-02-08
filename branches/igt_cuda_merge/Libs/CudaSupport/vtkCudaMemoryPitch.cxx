#include "vtkCudaMemoryPitch.h"
#include "cuda_runtime_api.h"
#include "vtkCudaBase.h"

#include "vtkObjectFactory.h"

vtkCxxRevisionMacro(vtkCudaMemoryPitch, "$Revision: 1.0 $");
vtkStandardNewMacro(vtkCudaMemoryPitch);

vtkCudaMemoryPitch::vtkCudaMemoryPitch()
{
    this->Pitch = 0;
}

vtkCudaMemoryPitch::~vtkCudaMemoryPitch()
{
    this->Free();
}


void* vtkCudaMemoryPitch::AllocatePitchBytes(size_t width, size_t height, size_t typeSize)
{
    this->Free();
    cudaError_t error = 
        cudaMallocPitch(&this->MemPointer, &this->Pitch, width * typeSize, height);
    this->Width = width;
    this->Height = height;
    if (error != cudaSuccess)
        vtkCudaBase::PrintError(error);

    return (void*)this->MemPointer;
}

void vtkCudaMemoryPitch::Free()
{  
    this->Superclass::Free();
    this->Pitch = 0;
}


void vtkCudaMemoryPitch::MemSet(int value)
{
    cudaMemset2D(this->MemPointer, this->Pitch, value, this->Width, this->Height);  
}

void vtkCudaMemoryPitch::PrintSelf(ostream &os, vtkIndent indent)
{
    this->Superclass::PrintSelf(os, indent);
    os << " Width: "<< this->GetWidth() << 
        " Height: " << this->GetHeight() <<
        " Pitch: " << this->GetPitch();
}
