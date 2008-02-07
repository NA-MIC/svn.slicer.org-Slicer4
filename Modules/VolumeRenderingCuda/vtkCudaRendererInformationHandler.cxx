#include "vtkCudaRendererInformationHandler.h"

// std
#include <vector>
// cuda functions
#include "vector_functions.h"

// vtk base
#include "vtkObjectFactory.h"

// Renderer Information
#include "vtkCamera.h"
#include "vtkRenderer.h"
#include "vtkRenderWindow.h"


vtkCxxRevisionMacro(vtkCudaRendererInformationHandler, "$Revision: 1.0 $");
vtkStandardNewMacro(vtkCudaRendererInformationHandler);

vtkCudaRendererInformationHandler::vtkCudaRendererInformationHandler()
{

}

vtkCudaRendererInformationHandler::~vtkCudaRendererInformationHandler()
{
    this->Renderer = NULL;

}


void vtkCudaRendererInformationHandler::SetRenderer(vtkRenderer* renderer)
{
    this->Renderer = renderer;
    this->Update();
}

// HACK
void vtkCudaRendererInformationHandler::SetZBuffer(Cudapp::DeviceMemory* ZBuffer)
{
    this->RendererInfo.ZBuffer = ZBuffer->GetMemPointerAs<float>();
}


void vtkCudaRendererInformationHandler::Update()
{
    if (this->Renderer != NULL)
    {
        // Renderplane Update.
        vtkRenderWindow *renWin= this->Renderer->GetRenderWindow();
        int *size=renWin->GetSize();
        this->RendererInfo.Resolution[0] = size[0];
        this->RendererInfo.Resolution[1] = size[1];


        vtkCamera* cam = this->Renderer->GetActiveCamera();

        // Update Lights.
        std::vector<float3> lights;
        lights.push_back(make_float3(0,0,1));

        this->RendererInfo.LightCount = lights.size();
        if (!lights.empty())
            this->RendererInfo.LightVectors = &lights[0];

        // Update Camera
        this->RendererInfo.CameraPos[0] = cam->GetPosition()[0];
        this->RendererInfo.CameraPos[1] = cam->GetPosition()[1];
        this->RendererInfo.CameraPos[2] = cam->GetPosition()[2];
        this->RendererInfo.TargetPos[0] = cam->GetFocalPoint()[0];
        this->RendererInfo.TargetPos[1] = cam->GetFocalPoint()[1];
        this->RendererInfo.TargetPos[2] = cam->GetFocalPoint()[2];
        this->RendererInfo.ViewUp[0] = cam->GetViewUp()[0];
        this->RendererInfo.ViewUp[1] = cam->GetViewUp()[1];
        this->RendererInfo.ViewUp[2] = cam->GetViewUp()[2];
        this->RendererInfo.NearPlane = -500;
        this->RendererInfo.FarPlane = 1000;
    }
}
