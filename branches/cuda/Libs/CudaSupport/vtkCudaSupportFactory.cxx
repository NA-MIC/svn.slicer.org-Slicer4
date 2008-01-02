#include "vtkCudaSupportFactory.h"
#include "vtkObjectFactory.h"

vtkCxxRevisionMacro(vtkCudaSupportFactory, "$Revision: 1.0 $");
vtkStandardNewMacro(vtkCudaSupportFactory);

vtkObject* vtkCudaSupportFactory::CreateInstance(const char* vtkclassname)
{
  vtkObject *ret = vtkObjectFactory::CreateInstance(vtkclassname);
  if (ret != NULL)
    return ret;  
  else 
    return NULL;
}
