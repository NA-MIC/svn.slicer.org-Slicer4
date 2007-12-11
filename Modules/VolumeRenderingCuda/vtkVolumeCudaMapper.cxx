#include "vtkVolumeCudaMapper.h"
#include "vtkVolumeRenderingCudaFactory.h"

#include <cutil.h>
#include <cuda_runtime_api.h>

vtkCxxRevisionMacro(vtkVolumeCudaMapper, "$Revision: 1.6 $");

//----------------------------------------------------------------------------
// Needed when we don't use the vtkStandardNewMacro.
vtkInstantiatorNewMacro(vtkVolumeCudaMapper);

vtkVolumeCudaMapper::vtkVolumeCudaMapper()
{

}

vtkVolumeCudaMapper::~vtkVolumeCudaMapper()
{
}

void vtkVolumeCudaMapper::PrintSelf(ostream& os, vtkIndent indent)
{
// this->SuperClass::PrintSelf();
}

vtkVolumeCudaMapper *vtkVolumeCudaMapper::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret =
      vtkVolumeRenderingCudaFactory::CreateInstance("vtkVolumeCUdaMapper");
  return (vtkVolumeCudaMapper*)ret;
}


int vtkVolumeCudaMapper::CheckSupportedCudaVersion(int cudaVersion)
{
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);

  int device;
  for (device = 0; device < deviceCount; ++device)
  {
    cudaDeviceProp deviceProperty;
    cudaGetDeviceProperties(&deviceProperty, device);
  }

  /// HACK
  return 0;
}

