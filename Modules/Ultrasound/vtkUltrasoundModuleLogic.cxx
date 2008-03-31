#include "vtkUltrasoundModuleLogic.h"
#include "vtkObjectFactory.h"
#include "vtkObject.h"

vtkUltrasoundModuleLogic::vtkUltrasoundModuleLogic(void)
{
}

vtkUltrasoundModuleLogic::~vtkUltrasoundModuleLogic(void)
{
}
vtkUltrasoundModuleLogic* vtkUltrasoundModuleLogic::New()
{
 // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkUltrasoundModuleLogic");
  if(ret)
    {
      return (vtkUltrasoundModuleLogic*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkUltrasoundModuleLogic;
}
void vtkUltrasoundModuleLogic::PrintSelf(std::ostream &os, vtkIndent indent)
{
    os<<indent<<"Print logic"<<endl;
}
