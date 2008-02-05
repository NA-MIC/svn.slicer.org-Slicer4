

#include "CudappDeviceMemory.h"
#include "CudappLocalMemory.h"

int main(int argc, char** argv)
{
  CudappLocalMemory* local = CudappLocalMemory::New();
  CudappDeviceMemory* dev = CudappDeviceMemory::New();
  
  CudappLocalMemory* dest = CudappLocalMemory::New();

  local->Allocate<int>(1000);
  dev->Allocate<int>(1000);
  dest->Allocate<int>(1000);
  
  local->MemSet(5);
  
  local->CopyTo(dev);
  dev->CopyTo(dest);
  
  for (unsigned int i = 0; i < local->GetSize(); i++)
    if (local->GetMemPointerAs<int>()[i] != dest->GetMemPointerAs<int>()[i])
      return -1;

  return 0;  
}
