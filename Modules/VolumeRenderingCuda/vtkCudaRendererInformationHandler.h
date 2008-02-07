#ifndef VTKCUDARENDERERINFORMATIONHANDLER_H_
#define VTKCUDARENDERERINFORMATIONHANDLER_H_

#include "vtkObject.h"
#include "vtkVolumeRenderingCudaModule.h"

class vtkRenderer;

//BTX
//class cudaRendererInformation;
#include "cudaRendererInformation.h"
#include "CudappDeviceMemory.h"
#include "CudappLocalMemory.h"
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



    //HACK
    //BTX
    void SetZBuffer(Cudapp::DeviceMemory* ZBuffer);
    //ETX


    virtual void Update();

protected:
    vtkCudaRendererInformationHandler();
    ~vtkCudaRendererInformationHandler();

    void UpdateResolution(unsigned int width, unsigned int height);
private:
    vtkCudaRendererInformationHandler& operator=(const vtkCudaRendererInformationHandler&); // not implemented
    vtkCudaRendererInformationHandler(const vtkCudaRendererInformationHandler&); // not implemented

private:
    vtkRenderer*             Renderer;
    //BTX
    cudaRendererInformation  RendererInfo;

    //Cudapp::LocalMemory      LocalZBuffer;
    //Cudapp::DeviceMemory     CudaZBuffer;
    //ETX
};

#endif /* VTKCUDARENDERERINFORMATIONHANDLER_H_ */
