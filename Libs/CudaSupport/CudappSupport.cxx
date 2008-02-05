#include "CudappSupport.h"
#include "CudappDevice.h"

#include <cutil.h>
#include <cuda_runtime_api.h>

vtkCxxRevisionMacro(CudappSupport, "$Revision: 1.6$");

//----------------------------------------------------------------------------
// Needed when we don't use the vtkStandardNewMacro.
vtkInstantiatorNewMacro(CudappSupport);

CudappSupport* CudappSupport::New()
{
    return new CudappSupport();
}

CudappSupport::CudappSupport()
{
    CheckSupportedCudaVersion();
}

CudappSupport::~CudappSupport()
{
    for (int i = 0; i < this->GetDeviceCount(); i++)
        this->Devices[i]->Delete();
}

int CudappSupport::CheckSupportedCudaVersion()
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    int device;
    for (device = 0; device < deviceCount; ++device)
    {
        CudappDevice* CudaDevice = CudappDevice::New();
        CudaDevice->SetDeviceNumber(device);

        this->Devices.push_back(CudaDevice);
    }

    /// HACK BY NOW
    return 0;
}

void CudappSupport::PrintSelf(ostream& os, vtkIndent indent)
{
    os << "Cuda Support Listing all Children: "<< std::endl;
    for (int i = 0; i < this->GetDeviceCount(); ++i)
    {
        this->Devices[i]->PrintSelf(os, indent.GetNextIndent());
    }
}
