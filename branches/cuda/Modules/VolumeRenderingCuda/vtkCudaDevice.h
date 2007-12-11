#ifndef VTKCUDADEVICE_H_
#define VTKCUDADEVICE_H_

#include "vtkObject.h"
#include "vtkVolumeRenderingCudaModule.h"

class VTK_VOLUMERENDERINGCUDAMODULE_EXPORT vtkCudaDevice : public vtkObject
{
  vtkTypeRevisionMacro(vtkCudaDevice, vtkObject);
        
  vtkGetMacro(Initialized, bool);
        
  static vtkCudaDevice* New();
        
  void PrintSelf(ostream& os, vtkIndent indent);
        
  protected:
    vtkCudaDevice();
    virtual ~vtkCudaDevice();

    bool Initialized;
};

#endif /*VTKCUDADEVICE_H_*/
