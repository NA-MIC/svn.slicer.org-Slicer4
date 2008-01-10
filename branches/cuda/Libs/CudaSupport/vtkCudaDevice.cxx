#include "vtkCudaDevice.h"
#include "cuda_runtime_api.h"
#include "vtkCudaMemoryArray.h"

#include "vtkObjectFactory.h"

vtkCxxRevisionMacro(vtkCudaDevice, "$Revision: 1.6 $");
vtkStandardNewMacro(vtkCudaDevice);

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
    return false;
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

void vtkCudaDevice::SynchronizeThread()
{
    cudaThreadSynchronize();
}
void vtkCudaDevice::ExitThread()
{

}


void vtkCudaDevice::PrintSelf(ostream& os, vtkIndent indent)
{
    this->Superclass::PrintSelf(os, indent);
    os << "Device Name = " << this->GetName() << 
        "DeviceNumber = " << this->GetDeviceNumber() <<
        "Version = " << this->GetMajor() << "." << this->GetMinor() <<
        "Memory: Global = " << this->GetTotalGlobalMem() <<
        " Shared Per Block = " << this->GetSharedMemPerBlock() <<
        " Regisers Per Block = " << this->GetRegsPerBlock() << 
        " Wrap Size " << this->GetWrapSize() <<
        " Pitch Size = " << this->GetMemPitch() <<
        " Threads Per Block = " << this->GetMaxThreadsPerBlock() <<
        " Max Threads Dimension = " << this->GetMaxThreadsDim()[0] << "x" << this->GetMaxThreadsDim()[1] << "x" << this->GetMaxThreadsDim()[2] <<
        " Max Grid Size = " << this->GetMaxGridSize()[0] << "x" << this->GetMaxGridSize()[1] << "x" << this->GetMaxGridSize()[2] <<
        " Total Constant Memory = " << this->GetTotalConstMem() << 
        " Clock Rate = " << this->GetClockRate() << "kHz" << 
        " Texture Alignment = " << this->GetTextureAlignment();
}
