#include "vtkCudaSupport.h"
#include "vtkCudaDevice.h"

#include <cutil.h>
#include <cuda_runtime_api.h>

vtkCxxRevisionMacro(vtkCudaSupport, "$Revision: 1.6$");

//----------------------------------------------------------------------------
// Needed when we don't use the vtkStandardNewMacro.
vtkInstantiatorNewMacro(vtkCudaSupport);

vtkCudaSupport* vtkCudaSupport::New()
{
    return new vtkCudaSupport();
}

vtkCudaSupport::vtkCudaSupport()
{
    CheckSupportedCudaVersion();
}

vtkCudaSupport::~vtkCudaSupport()
{
    for (int i = 0; i < this->GetDeviceCount(); i++)
        this->Devices[i]->Delete();
}

int vtkCudaSupport::CheckSupportedCudaVersion()
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    int device;
    for (device = 0; device < deviceCount; ++device)
    {
        vtkCudaDevice* CudaDevice = vtkCudaDevice::New();
        CudaDevice->SetDeviceNumber(device);

        this->Devices.push_back(CudaDevice);
    }

    /// HACK BY NOW
    return 0;
}

void vtkCudaSupport::PrintSelf(ostream& os, vtkIndent indent)
{
    os << "Cuda Support Listing all Children: "<< std::endl;
    for (int i = 0; i < this->GetDeviceCount(); ++i)
    {
        this->Devices[i]->PrintSelf(os, indent.GetNextIndent());
    }
}
