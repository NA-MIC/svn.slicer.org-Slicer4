#ifndef VTKCUDARENDERERINFORMATIONHANDLER_H_
#define VTKCUDARENDERERINFORMATIONHANDLER_H_

#include "vtkObject.h"
#include "vtkVolumeRenderingCudaModule.h"

class vtkRenderer;
//BTX
//class cudaRendererInformation;
#include "cudaRendererInformation.h"
//ETX
class VTK_VOLUMERENDERINGCUDAMODULE_EXPORT vtkCudaRendererInformationHandler : public vtkObject
{
    vtkTypeRevisionMacro(vtkCudaRendererInformationHandler, vtkObject);
public:
    static vtkCudaRendererInformationHandler* New();

    //BTX
    void SetRenderer(vtkRenderer* renderer);
    vtkGetMacro(Renderer, vtkRenderer*);
    const cudaRendererInformation& GetRendererInfo() { return this->RendererInfo; }
    //ETX

    virtual void Update();

protected:
    vtkCudaRendererInformationHandler();
    ~vtkCudaRendererInformationHandler();
private:
    vtkCudaRendererInformationHandler& operator=(const vtkCudaRendererInformationHandler&); // not implemented
    vtkCudaRendererInformationHandler(const vtkCudaRendererInformationHandler&); // not implemented

    vtkRenderer*             Renderer;
    //BTX
    cudaRendererInformation  RendererInfo;
    //ETX
};

#endif /* VTKCUDARENDERERINFORMATIONHANDLER_H_ */
