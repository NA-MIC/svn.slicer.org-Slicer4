// TYPE
#include "vtkCudaVolumeProperty.h"
#include "vtkObjectFactory.h"



vtkCxxRevisionMacro(vtkVolumeCudaMapper, "$Revision: 1.6 $");
vtkStandardNewMacro(vtkVolumeCudaMapper);

vtkCudaVolumeProperty *vtkCudaVolumeProperty::New()
{
  return vtkCudaVolumeProperty::New();  
}


vtkCudaVolumeProperty::vtkCudaVolumeProperty()
{
}

vtkCudaVolumeProperty::~vtkCudaVolumeProperty()
{
}
