#include "CudappHostMemory.h"

#include "CudappBase.h"
#include "vtkObjectFactory.h"

#include <string.h>
#include "cuda_runtime_api.h"


vtkCxxRevisionMacro(CudappHostMemory, "$Revision 1.0 $");
vtkStandardNewMacro(CudappHostMemory);

CudappHostMemory::CudappHostMemory()
{
}

CudappHostMemory::~CudappHostMemory()
{
    this->Free();
}


void* CudappHostMemory::AllocateBytes(size_t count)
{
    this->Free();
    cudaError_t error = 
        cudaMallocHost(&this->MemPointer, count);
    this->Size = count;
    if (error != cudaSuccess)
        CudappBase::PrintError(error);

    return (void*) this->MemPointer;
}

void CudappHostMemory::Free()
{
    if (this->MemPointer != NULL)
    {
        cudaFreeHost(this->MemPointer);
        this->MemPointer = NULL;  
        this->Size = 0;
    }
}

void CudappHostMemory::PrintSelf(ostream& os, vtkIndent indent)
{
    this->Superclass::PrintSelf(os, indent);
}
