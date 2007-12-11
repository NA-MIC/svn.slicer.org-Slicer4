#include "vtkCudaDevice.h"

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
}

vtkCudaDevice::~vtkCudaDevice()
{
}

void vtkCudaDevice::PrintSelf(ostream& os, vtkIndent indent)
{

}
