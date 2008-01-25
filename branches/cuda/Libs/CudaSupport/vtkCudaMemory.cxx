#include "vtkCudaMemory.h"

#include "cuda_runtime_api.h"
#include "vtkCudaBase.h"

#include "vtkCudaLocalMemory.h"
#include "vtkCudaHostMemory.h"
#include "vtkCudaMemoryArray.h"

#include "vtkObjectFactory.h"
#include "vtkImageData.h"

vtkCxxRevisionMacro(vtkCudaMemory, "$Revision: 1.0 $");

vtkCudaMemory::vtkCudaMemory()
{
    this->MemPointer = NULL;
    this->Size = 0;
}

vtkCudaMemory::~vtkCudaMemory()
{
    // so the virtual function call will not be false.
    // each subclass must call free by its own and set MemPointer to NULL in its Destructor!
    //if (this->MemPointer != NULL)
    //    this->Free();
}

void vtkCudaMemory::PrintSelf (ostream &os, vtkIndent indent)
{
    this->Superclass::PrintSelf(os, indent);
    if (this->GetMemPointer() == NULL)
        os << "Not yet allocated";
}


bool vtkCudaMemory::CopyFrom(vtkCudaMemory* mem)
{
  return this->CopyFrom(mem->GetMemPointer(), mem->GetSize(), (size_t)0, mem->GetMemoryLocation());  
}
