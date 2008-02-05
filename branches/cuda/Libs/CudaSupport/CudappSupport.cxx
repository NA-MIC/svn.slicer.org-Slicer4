#include "CudappSupport.h"
#include "CudappDevice.h"

#include <cutil.h>
#include <cuda_runtime_api.h>

CudappSupport::CudappSupport()
{
    CheckSupportedCudaVersion();
}

CudappSupport::~CudappSupport()
{
    for (int i = 0; i < this->GetDeviceCount(); i++)
        delete this->Devices[i];
}

int CudappSupport::CheckSupportedCudaVersion()
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    int device;
    for (device = 0; device < deviceCount; ++device)
    {
        CudappDevice* CudaDevice = new CudappDevice;
        CudaDevice->SetDeviceNumber(device);

        this->Devices.push_back(CudaDevice);
    }

    /// HACK BY NOW
    return 0;
}

void CudappSupport::PrintSelf(std::ostream&  os)
{
    os << "Cuda Support Listing all Children: "<< std::endl;
    for (int i = 0; i < this->GetDeviceCount(); ++i)
    {
        this->Devices[i]->PrintSelf(os);
    }
}
