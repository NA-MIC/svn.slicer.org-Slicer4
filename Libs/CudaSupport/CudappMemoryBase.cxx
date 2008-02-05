#include "CudappMemoryBase.h"

#include "CudappMemory.h"

CudappMemoryBase::CudappMemoryBase()
{
  this->Location = CudappMemoryBase::MemoryOnHost;
}

CudappMemoryBase::~CudappMemoryBase()
{
}

void CudappMemoryBase::PrintSelf (std::ostream &os)
{
    os << "Size = " << this->GetSize();  
}
