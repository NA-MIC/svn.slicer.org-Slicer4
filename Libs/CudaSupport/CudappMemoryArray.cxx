#include "CudappMemoryArray.h"
#include "CudappBase.h"

CudappMemoryArray::CudappMemoryArray()
{
  this->Location = CudappMemoryBase::MemoryOnDevice;

    this->Array = NULL;
    this->Width = this->Height = 0;
    this->Descriptor.x = this->Descriptor.y = this->Descriptor.z = 0;
    this->Descriptor.f = cudaChannelFormatKindSigned;
}

CudappMemoryArray::~CudappMemoryArray()
{
    this->Free();
}

/**
* Allocates A new array of size width*height with the specified type T
* @param width the width of the array to allocate
* @param height the height of the array to allocate.
* 
* @note if there was already allocated data in this instance the data will be erased.
*/
void CudappMemoryArray::Allocate(size_t width, size_t height)
{
    this->Free();

    cudaMallocArray(&this->Array, &this->Descriptor, width, height);
    this->Width = width;
    this->Height = height;
    this->Size = width*height * /* HACK */ sizeof(uchar4);
}

/**
* @brief frees all the resources needed for the Array
*/
void CudappMemoryArray::Free()
{
    if (this->Array != NULL) {
        cudaFreeArray(this->Array);  
        this->Array = NULL;
        this->Width = this->Height = 0;
        this->Size = 0;
    }
}

void CudappMemoryArray::DeepCopy(CudappMemoryArray* source)
{
    this->Allocate(source->GetWidth(), source->GetHeight());
    cudaMemcpyArrayToArray(this->Array, 0, 0, source->Array, 0, 0, sizeof(source->Array));
}


void CudappMemoryArray::PrintSelf(ostream &os)
{
    this->Superclass::PrintSelf(os, indent);
    os << " Width: "<< this->GetWidth() << 
        " Height: " << this->GetHeight();
}
