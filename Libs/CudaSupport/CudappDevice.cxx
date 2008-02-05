#include "CudappDevice.h"
#include "cuda_runtime_api.h"
#include "CudappMemoryArray.h"

CudappDevice::CudappDevice()
{
    this->Initialized = false;
    this->DeviceNumber = 0;

    // set the device properties to a 'don't care' value
    cudaDeviceProp prop = cudaDevicePropDontCare;
    this->DeviceProp = prop;
}

CudappDevice::~CudappDevice()
{
}

/**
* TODO Remove this function. just for trial
*/
bool CudappDevice::AllocateMemory()
{
    return false;
}

void CudappDevice::SetDeviceNumber(int deviceNumber)
{
    this->DeviceNumber = deviceNumber;
    this->LoadDeviceProperties();
}

void CudappDevice::LoadDeviceProperties()
{
    cudaGetDeviceProperties(&DeviceProp, this->DeviceNumber);
}

void CudappDevice::MakeActive()
{
    cudaSetDevice(this->DeviceNumber);
    this->Initialized = true;
}

void CudappDevice::SynchronizeThread()
{
    cudaThreadSynchronize();
}
void CudappDevice::ExitThread()
{

}

void CudappDevice::PrintSelf(std::ostream&  os)
{
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
