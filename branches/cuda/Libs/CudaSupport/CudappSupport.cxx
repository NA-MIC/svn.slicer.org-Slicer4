#include "CudappSupport.h"
#include "CudappDevice.h"

#include <cutil.h>
#include <cuda_runtime_api.h>

namespace Cudapp
{
    Support::Support()
    {
        CheckSupportedCudaVersion();
    }

    Support::~Support()
    {
        for (int i = 0; i < this->GetDeviceCount(); i++)
            delete this->Devices[i];
    }

    int Support::CheckSupportedCudaVersion()
    {
        int deviceCount = 0;
        cudaGetDeviceCount(&deviceCount);
        int device;
        for (device = 0; device < deviceCount; ++device)
        {
            Device* CudaDevice = new Device;
            CudaDevice->SetDeviceNumber(device);

            this->Devices.push_back(CudaDevice);
        }

        /// HACK BY NOW
        return 0;
    }

    void Support::PrintSelf(std::ostream&  os)
    {
        os << "Cuda Support Listing all Children: "<< std::endl;
        for (int i = 0; i < this->GetDeviceCount(); ++i)
        {
            this->Devices[i]->PrintSelf(os);
        }
    }
}
