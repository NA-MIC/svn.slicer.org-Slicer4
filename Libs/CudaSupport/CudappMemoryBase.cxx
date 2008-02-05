#include "CudappMemoryBase.h"

#include "CudappMemory.h"

CudappMemoryBase* CudappMemoryBase::New()
{
    return CudappMemory::New();
}

CudappMemoryBase::CudappMemoryBase()
{
  this->Location = CudappMemoryBase::MemoryOnHost;
}

CudappMemoryBase::~CudappMemoryBase()
{
}

void CudappMemoryBase::PrintSelf (ostream &os)
{
    this->Superclass::PrintSelf(os, indent);
    os << "Size = " << this->GetSize();  
}
