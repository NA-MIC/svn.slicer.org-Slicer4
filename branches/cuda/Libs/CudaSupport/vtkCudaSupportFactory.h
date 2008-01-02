#ifndef VTKSUPPORTFACTORY_H_
#define VTKSUPPORTFACTORY_H_

#include "vtkObject.h"
#include "vtkCudaSupportModule.h"

class VTK_CUDASUPPORT_EXPORT vtkCudaSupportFactory : public vtkObject
{
  vtkTypeRevisionMacro(vtkCudaSupportFactory, vtkObject);
  static vtkCudaSupportFactory* New();
   // Description:
     // Create and return an instance of the named vtk object.
     // This method first checks the vtkObjectFactory to support
     // dynamic loading. 
  static vtkObject* CreateInstance(const char* vtkclassname);
  
protected:
  vtkCudaSupportFactory() {};
  virtual ~vtkCudaSupportFactory() {};
  
  vtkCudaSupportFactory(const vtkCudaSupportFactory&);
  vtkCudaSupportFactory& operator=(const vtkCudaSupportFactory&);
};

#endif /*VTKSUPPORTFACTORY_H_*/
