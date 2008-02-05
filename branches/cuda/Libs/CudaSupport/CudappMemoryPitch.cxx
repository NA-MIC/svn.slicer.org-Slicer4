#include "CudappMemoryPitch.h"
#include "cuda_runtime_api.h"
#include "CudappBase.h"

#include "vtkObjectFactory.h"

vtkCxxRevisionMacro(CudappMemoryPitch, "$Revision: 1.0 $");
vtkStandardNewMacro(CudappMemoryPitch);

CudappMemoryPitch::CudappMemoryPitch()
{
  this->Location = CudappMemoryBase::MemoryOnDevice;
  this->Location = CudappMemoryBase::MemoryOnDevice;
  
    this->Pitch = 0;
}

CudappMemoryPitch::~CudappMemoryPitch()
{
    this->Free();
}


void* CudappMemoryPitch::AllocatePitchBytes(size_t width, size_t height, size_t typeSize)
{
    this->Free();
    cudaError_t error = 
        cudaMallocPitch(&this->MemPointer, &this->Pitch, width * typeSize, height);
    this->Width = width;
    this->Height = height;
    if (error != cudaSuccess)
        CudappBase::PrintError(error);

    return (void*)this->MemPointer;
}



void CudappMemoryPitch::Free()
{  
    if (this->MemPointer != NULL)
    {
        cudaFree(this->MemPointer);  
        this->MemPointer = NULL;
        this->Size = 0;
        this->Pitch = 0;
    }
}


void CudappMemoryPitch::MemSet(int value)
{
    cudaMemset2D(this->MemPointer, this->Pitch, value, this->Width, this->Height);  
}

void CudappMemoryPitch::PrintSelf(ostream &os, vtkIndent indent)
{
    this->Superclass::PrintSelf(os, indent);
    os << " Width: "<< this->GetWidth() << 
        " Height: " << this->GetHeight() <<
        " Pitch: " << this->GetPitch();
}
