#include "CudappMemory.h"

#include "cuda_runtime_api.h"
#include "CudappBase.h"

#include "CudappLocalMemory.h"
#include "CudappHostMemory.h"
#include "CudappMemoryArray.h"

namespace Cudapp
{
    Memory::Memory()
    {
        this->MemPointer = NULL;
        this->Size = 0;
    }

    Memory::~Memory()
    {
        // so the virtual function call will not be false.
        // each subclass must call free by its own and set MemPointer to NULL in its Destructor!
        //if (this->MemPointer != NULL)
        //    this->Free();
    }

    void Memory::PrintSelf (std::ostream &os) const
    {
        this->MemoryBase::PrintSelf(os);
        if (this->GetMemPointer() == NULL)
            os << "Not yet allocated";
    }


    bool Memory::CopyFrom(Memory* mem)
    {
        return this->CopyFrom(mem->GetMemPointer(), mem->GetSize(), (size_t)0, mem->GetMemoryLocation());  
    }
}
