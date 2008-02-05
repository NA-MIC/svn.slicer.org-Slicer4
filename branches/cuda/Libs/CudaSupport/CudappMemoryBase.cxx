#include "CudappMemoryBase.h"

#include "CudappMemory.h"

vtkCxxRevisionMacro(CudappMemoryBase, "$Revision: 1.0$");

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

void CudappMemoryBase::PrintSelf (ostream &os, vtkIndent indent)
{
    this->Superclass::PrintSelf(os, indent);
    os << "Size = " << this->GetSize();  
}
