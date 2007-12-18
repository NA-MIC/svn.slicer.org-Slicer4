#include "vtkCudaDevice.h"
#include "cuda_runtime_api.h"
#include "vtkCudaMemoryArray.h"

vtkCxxRevisionMacro(vtkCudaDevice, "$Revision: 1.6 $");

//----------------------------------------------------------------------------
// Needed when we don't use the vtkStandardNewMacro.
vtkInstantiatorNewMacro(vtkCudaDevice);

vtkCudaDevice* vtkCudaDevice::New()
{
    /// HACK
    return new vtkCudaDevice();
}


vtkCudaDevice::vtkCudaDevice()
{
  this->Initialized = false;
  this->DeviceNumber = 0;
  
  // set the device properties to a 'don't care' value
  cudaDeviceProp prop = cudaDevicePropDontCare;
  this->DeviceProp = prop;
}

vtkCudaDevice::~vtkCudaDevice()
{
}

/**
 * TODO Remove this function. just for trial
 */
bool vtkCudaDevice::AllocateMemory()
{
  vtkCudaMemoryArray<float>* array = vtkCudaMemoryArray<float>::New();
}

void vtkCudaDevice::SetDeviceNumber(int deviceNumber)
{
  this->DeviceNumber = deviceNumber;
  this->LoadDeviceProperties();
}

void vtkCudaDevice::LoadDeviceProperties()
{
  cudaGetDeviceProperties(&DeviceProp, this->DeviceNumber);
}

void vtkCudaDevice::MakeActive()
{
  cudaSetDevice(this->DeviceNumber);
  this->Initialized = true;
}

void vtkCudaDevice::PrintSelf(ostream& os, vtkIndent indent)
{
  os << this->GetName();
}
