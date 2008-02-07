#ifndef vtkCudaVolumeInformationHandler_H_
#define vtkCudaVolumeInformationHandler_H_

#include "vtkObject.h"
#include "vtkVolumeRenderingCudaModule.h"

class VTK_VOLUMERENDERINGCUDAMODULE_EXPORT vtkCudaVolumeInformationHandler : public vtkObject
{
    vtkTypeRevisionMacro(vtkCudaVolumeInformationHandler, vtkObject);
public:
    static vtkCudaVolumeInformationHandler* New();

protected:
    vtkCudaVolumeInformationHandler();
    ~vtkCudaVolumeInformationHandler();

private:
    vtkCudaVolumeInformationHandler& operator=(const vtkCudaVolumeInformationHandler&); // not implemented
    vtkCudaVolumeInformationHandler(const vtkCudaVolumeInformationHandler&); // not implemented
};

#endif /* vtkCudaVolumeInformationHandler_H_ */
