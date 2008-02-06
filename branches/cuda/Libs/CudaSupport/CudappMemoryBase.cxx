#include "CudappMemoryBase.h"

namespace Cudapp
{
    MemoryBase::MemoryBase()
    {
        this->Location = MemoryBase::MemoryOnHost;
    }

    MemoryBase::~MemoryBase()
    {
    }

    void MemoryBase::PrintSelf (std::ostream &os)
    {
        os << "Size = " << this->GetSize();  
    }
}
