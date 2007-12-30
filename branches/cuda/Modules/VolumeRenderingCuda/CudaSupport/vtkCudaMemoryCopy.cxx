#include "vtkCudaMemoryCopy.h"

#include "vtkCudaMemory.h"
#include "vtkCudaHostMemory.h"
#include "vtkCudaMemoryArray.h"
#include "vtkCudaMemoryPitch.h"

#include "cuda_runtime_api.h"

vtkCudaMemoryCopy* vtkCudaMemoryCopy::New()
{
  return new vtkCudaMemoryCopy();  
}

vtkCudaMemoryCopy::vtkCudaMemoryCopy()
{
  this->Source = NULL;
  this->Destination = NULL;
}

vtkCudaMemoryCopy::~vtkCudaMemoryCopy()
{
}

vtkCudaMemoryBase::MemoryType vtkCudaMemoryCopy::GetSourceType() const
{
  if (Source != NULL)
    return Source->GetType();
  else
    return vtkCudaMemoryBase::Undefined;  
}

vtkCudaMemory* vtkCudaMemoryCopy::CopyToMemory()
{
  vtkCudaMemory* dest = NULL;
  switch (this->GetSourceType())
  {
    default:
       break;
    case vtkCudaMemoryBase::Memory:
      //Copy Memory to Memory:
      dest = vtkCudaMemory::New();
      dest->AllocateBytes(this->Source->GetSize());
      cudaMemcpy(dest->GetMemPointer(), 
          ((vtkCudaMemory*)this->Source)->GetMemPointer(),
          this->Source->GetSize(),
          cudaMemcpyDeviceToDevice
          );
      break;
    case vtkCudaMemoryBase::HostMemory:
      dest = vtkCudaMemory::New();
      dest->AllocateBytes(this->Source->GetSize());
      cudaMemcpy(dest->GetMemPointer(), 
          ((vtkCudaMemory*)this->Source)->GetMemPointer(),
          this->Source->GetSize(),
          cudaMemcpyHostToDevice
          );
      break;
    case vtkCudaMemoryBase::ArrayMemory:
      dest = vtkCudaMemory::New();
      dest->AllocateBytes(this->Source->GetSize());
      cudaMemcpyFromArray(dest->GetMemPointer(), ((vtkCudaMemoryArray*)this->Source)->GetArray(), 
      0, 0,this->Source->GetSize(),
      cudaMemcpyDeviceToDevice);
      break;
    case vtkCudaMemoryBase::PitchMemory:

      break;
    
  }
  return dest;
}

vtkCudaHostMemory* vtkCudaMemoryCopy::CopyToHostMemory()
{
    vtkCudaHostMemory* retVal = NULL;
  switch (this->GetSourceType())
  {
    default:
       break;
    case vtkCudaMemoryBase::Memory:
      
      break;
    case vtkCudaMemoryBase::HostMemory:
      
      break;
    case vtkCudaMemoryBase::ArrayMemory:
      
      break;
    case vtkCudaMemoryBase::PitchMemory:
      
      break;
    
  }
  return retVal;
}

vtkCudaMemoryPitch* vtkCudaMemoryCopy::CopyToMemoryPitch()
{
    vtkCudaMemoryPitch* retVal = NULL;
  switch (this->GetSourceType())
  {
    default:
       break;
    case vtkCudaMemoryBase::Memory:
      
      break;
    case vtkCudaMemoryBase::HostMemory:
      
      break;
    case vtkCudaMemoryBase::ArrayMemory:
      
      break;
    case vtkCudaMemoryBase::PitchMemory:
      
      break;
    
  }
  return retVal;
}

vtkCudaMemoryArray* vtkCudaMemoryCopy::CopyToMemoryArray()
{
    vtkCudaMemoryArray* retVal = NULL;
  switch (this->GetSourceType())
  {
    default:
       break;
    case vtkCudaMemoryBase::Memory:
      
      break;
    case vtkCudaMemoryBase::HostMemory:
      
      break;
    case vtkCudaMemoryBase::ArrayMemory:
      
      break;
    case vtkCudaMemoryBase::PitchMemory:
      
      break;
    
  }
  return retVal;
}
