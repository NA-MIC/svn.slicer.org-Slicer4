#include "vtkCudaMemoryBase.h"

#include "vtkCudaMemory.h"

vtkCxxRevisionMacro(vtkCudaMemoryBase, "$Revision: 1.0$");

vtkCudaMemoryBase* vtkCudaMemoryBase::New()
{
    return vtkCudaMemory::New();
}

vtkCudaMemoryBase::~vtkCudaMemoryBase()
{
}

vtkCudaMemoryBase::vtkCudaMemoryBase()
{
}

void vtkCudaMemoryBase::PrintSelf (ostream &os, vtkIndent indent)
{
    this->Superclass::PrintSelf(os, indent);
    os << "Size = " << this->GetSize();  
}
