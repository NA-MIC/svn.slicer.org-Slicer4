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
        if (size[0] != this->RendererInfo.Resolution[0] ||
            size[1] != this->RendererInfo.Resolution[1])
        {
            this->RendererInfo.Resolution[0] = size[0];
            this->RendererInfo.Resolution[1] = size[1];

            // HACK -> Allocate is too slow!!
            LocalZBuffer.Allocate<float>(this->RendererInfo.Resolution[0] * this->RendererInfo.Resolution[1]);
            CudaZBuffer.Allocate<float>(this->RendererInfo.Resolution[0] * this->RendererInfo.Resolution[1]);
        }
        for (unsigned int i = 0 ; i < this->RendererInfo.Resolution[0] * this->RendererInfo.Resolution[1]; i++)
            this->LocalZBuffer.GetMemPointerAs<float>()[i] = 100000;

        //renderer->GetRenderWindow()->GetZbufferData(0,0,this->OutputDataSize[0]-1, this->OutputDataSize[1]-1, this->LocalZBuffer->GetMemPointerAs<float>());
        this->LocalZBuffer.CopyTo(&this->CudaZBuffer);
        this->RendererInfo.ZBuffer = CudaZBuffer.GetMemPointerAs<float>();

        this->MemoryTexture->SetSize(this->RendererInfo.Resolution[0], this->RendererInfo.Resolution[1]);
        this->RendererInfo.OutputImage = (uchar4*)this->MemoryTexture->GetRenderDestination();

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
        double clipRange[2];
        cam->GetClippingRange(clipRange);
        this->RendererInfo.ClippingRange[0] = (float)clipRange[0];
        this->RendererInfo.ClippingRange[1] = (float)clipRange[1];
    }
}
