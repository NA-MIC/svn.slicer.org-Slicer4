#include "vtkCudaSupport.h"

#include "vtkVolumeRenderingCudaFactory.h"

#include <cutil.h>
#include <cuda_runtime_api.h>

vtkCxxRevisionMacro(vtkCudaSupport, "$Revision: 1.6 $");

//----------------------------------------------------------------------------
// Needed when we don't use the vtkStandardNewMacro.
vtkInstantiatorNewMacro(vtkCudaSupport);

vtkCudaSupport::vtkCudaSupport()
{
}

vtkCudaSupport::~vtkCudaSupport()
{
}

int vtkCudaSupport::CheckSupportedCudaVersion(int cudaVersion)
{
        int deviceCount = 0;
        cudaGetDeviceCount(&deviceCount);
        int device;
        for (device = 0; device < deviceCount; ++device)
        {
                cudaDeviceProp deviceProperty;
                cudaGetDeviceProperties(&deviceProperty, device);
        }
}
