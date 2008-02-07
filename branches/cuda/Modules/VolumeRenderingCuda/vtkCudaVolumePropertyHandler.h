#ifndef VTKCUDAVOLUMEPROPERTYHANDLER_H_
#define VTKCUDAVOLUMEPROPERTYHANDLER_H_

#include "vtkObject.h"
#include "vtkVolumeRenderingCudaModule.h"

class VTK_VOLUMERENDERINGCUDAMODULE_EXPORT vtkCudaVolumePropertyHandler : public vtkObject
{
    vtkTypeRevisionMacro(vtkCudaVolumePropertyHandler, vtkObject);
public:
    static vtkCudaVolumePropertyHandler* New();

protected:
    vtkCudaVolumePropertyHandler();
    ~vtkCudaVolumePropertyHandler();
private:
    vtkCudaVolumePropertyHandler& operator=(const vtkCudaVolumePropertyHandler&); // not implemented
    vtkCudaVolumePropertyHandler(const vtkCudaVolumePropertyHandler&); // not implemented
};

#endif /* VTKCUDAVOLUMEPROPERTYHANDLER_H_ */
