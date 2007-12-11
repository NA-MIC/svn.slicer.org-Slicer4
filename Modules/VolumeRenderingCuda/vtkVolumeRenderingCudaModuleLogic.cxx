#include "vtkVolumeRenderingCudaModuleLogic.h"
#include "vtkObjectFactory.h"
#include "vtkObject.h"

vtkVolumeRenderingCudaModuleLogig* vtkVolumeRenderingCudaModuleLogic::New()
{
///HACK
    return new vtkVolumeRenderingCudaModuleLogic();
}

vtkVolumeRenderingCudaModuleLogic::vtkVolumeRenderingCudaModuleLogic(void)
{
}

vtkVolumeRenderingCudaModuleLogic::~vtkVolumeRenderingCudaModuleLogic(void)
{
}
vtkVolumeRenderingCudaModuleLogic* vtkVolumeRenderingCudaModuleLogic::New()
{
 // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkVolumeRenderingCudaModuleLogic");
  if(ret)
    {
      return (vtkVolumeRenderingCudaModuleLogic*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkVolumeRenderingCudaModuleLogic;
}
void vtkVolumeRenderingCudaModuleLogic::PrintSelf(std::ostream &os, vtkIndent indent)
{
    os<<indent<<"Print logic"<<endl;
}
