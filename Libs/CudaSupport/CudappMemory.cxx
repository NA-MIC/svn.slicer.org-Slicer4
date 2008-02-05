#include "CudappMemory.h"

#include "cuda_runtime_api.h"
#include "CudappBase.h"

#include "CudappLocalMemory.h"
#include "CudappHostMemory.h"
#include "CudappMemoryArray.h"


CudappMemory::CudappMemory()
{
    this->MemPointer = NULL;
    this->Size = 0;
}

CudappMemory::~CudappMemory()
{
    // so the virtual function call will not be false.
    // each subclass must call free by its own and set MemPointer to NULL in its Destructor!
    //if (this->MemPointer != NULL)
    //    this->Free();
}

void CudappMemory::PrintSelf (ostream &os)
{
    this->Superclass::PrintSelf(os, indent);
    if (this->GetMemPointer() == NULL)
        os << "Not yet allocated";
}


bool CudappMemory::CopyFrom(CudappMemory* mem)
{
  return this->CopyFrom(mem->GetMemPointer(), mem->GetSize(), (size_t)0, mem->GetMemoryLocation());  
}
