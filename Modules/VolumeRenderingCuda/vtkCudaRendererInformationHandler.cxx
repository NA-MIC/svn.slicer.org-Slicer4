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

// vtkCuda
#include "vtkCudaMemoryTexture.h"
vtkCxxRevisionMacro(vtkCudaRendererInformationHandler, "$Revision: 1.0 $");
vtkStandardNewMacro(vtkCudaRendererInformationHandler);

vtkCudaRendererInformationHandler::vtkCudaRendererInformationHandler()
{
    this->Renderer = NULL;
    this->MemoryTexture = vtkCudaMemoryTexture::New();
}

vtkCudaRendererInformationHandler::~vtkCudaRendererInformationHandler()
{
    this->Renderer = NULL;
    this->MemoryTexture->Delete();
}


void vtkCudaRendererInformationHandler::SetRenderer(vtkRenderer* renderer)
{
    this->Renderer = renderer;
    this->Update();
}

void vtkCudaRendererInformationHandler::Bind()
{
    this->MemoryTexture->BindTexture();
    this->MemoryTexture->BindBuffer();
    this->RendererInfo.OutputImage = (uchar4*)this->MemoryTexture->GetRenderDestination();
}

void vtkCudaRendererInformationHandler::Unbind()
{
    this->MemoryTexture->UnbindBuffer();
    this->MemoryTexture->UnbindTexture();
}

void vtkCudaRendererInformationHandler::Update()
{
    if (this->Renderer != NULL)
    {
        // Renderplane Update.
        vtkRenderWindow *renWin= this->Renderer->GetRenderWindow();
        int *size=renWin->GetSize();
        if (size[0] != this->RendererInfo.Resolution.x ||
            size[1] != this->RendererInfo.Resolution.y)
        {
            this->RendererInfo.Resolution.x = size[0];
            this->RendererInfo.Resolution.y = size[1];

            // HACK -> Allocate is too slow!!
            LocalZBuffer.Allocate<float>(this->RendererInfo.Resolution.x * this->RendererInfo.Resolution.y);
            CudaZBuffer.Allocate<float>(this->RendererInfo.Resolution.x * this->RendererInfo.Resolution.y);
        }
        for (unsigned int i = 0 ; i < this->RendererInfo.Resolution.x * this->RendererInfo.Resolution.y; i++)
            this->LocalZBuffer.GetMemPointerAs<float>()[i] = 100000;

        //renWin->GetZbufferData(0,0,this->RendererInfo.Resolution.x-1, this->RendererInfo.Resolution.y-1, this->LocalZBuffer.GetMemPointerAs<float>());
        this->LocalZBuffer.CopyTo(&this->CudaZBuffer);
        this->RendererInfo.ZBuffer = CudaZBuffer.GetMemPointerAs<float>();

        this->MemoryTexture->SetSize(this->RendererInfo.Resolution.x, this->RendererInfo.Resolution.y);
        this->RendererInfo.OutputImage = (uchar4*)this->MemoryTexture->GetRenderDestination();

        vtkCamera* cam = this->Renderer->GetActiveCamera();

        // Update Lights.
        std::vector<float3> lights;
        lights.push_back(make_float3(0,0,1));

        this->RendererInfo.LightCount = lights.size();
        if (!lights.empty())
            this->RendererInfo.LightVectors = &lights[0];

        // Update Camera
        this->RendererInfo.CameraPos.x = cam->GetPosition()[0];
        this->RendererInfo.CameraPos.y = cam->GetPosition()[1];
        this->RendererInfo.CameraPos.z = cam->GetPosition()[2];
        this->RendererInfo.TargetPos.x = cam->GetFocalPoint()[0];
        this->RendererInfo.TargetPos.y = cam->GetFocalPoint()[1];
        this->RendererInfo.TargetPos.z = cam->GetFocalPoint()[2];
        this->RendererInfo.ViewUp.x = cam->GetViewUp()[0];
        this->RendererInfo.ViewUp.y = cam->GetViewUp()[1];
        this->RendererInfo.ViewUp.z = cam->GetViewUp()[2];
        this->RendererInfo.CameraDirection.x= cam->GetDirectionOfProjection()[0];
        this->RendererInfo.CameraDirection.y= cam->GetDirectionOfProjection()[1];
        this->RendererInfo.CameraDirection.z= cam->GetDirectionOfProjection()[2];
        
        double clipRange[2];
        cam->GetClippingRange(clipRange);
        this->RendererInfo.ClippingRange.x = (float)clipRange[0];
        this->RendererInfo.ClippingRange.y = (float)clipRange[1];
    }
}
