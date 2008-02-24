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
#include "vtkMatrix4x4.h"

// vtkCuda
#include "vtkCudaMemoryTexture.h"
vtkCxxRevisionMacro(vtkCudaRendererInformationHandler, "$Revision: 1.0 $");
vtkStandardNewMacro(vtkCudaRendererInformationHandler);

vtkCudaRendererInformationHandler::vtkCudaRendererInformationHandler()
{
    this->Renderer = NULL;
    this->RendererInfo.ActualResolution.x = this->RendererInfo.ActualResolution.y = 0;
    this->MemoryTexture = vtkCudaMemoryTexture::New();

    this->SetRenderOutputScaleFactor(1.0f);


    this->mat = vtkMatrix4x4::New();
    this->mat->Identity();
    this->mat->Invert();
}

vtkCudaRendererInformationHandler::~vtkCudaRendererInformationHandler()
{
    this->Renderer = NULL;
    this->MemoryTexture->Delete();
    this->mat->Delete();
}


void vtkCudaRendererInformationHandler::SetRenderer(vtkRenderer* renderer)
{
    this->Renderer = renderer;
    this->Update();
}

void vtkCudaRendererInformationHandler::SetRenderOutputScaleFactor(float scaleFactor) 
{
    this->RenderOutputScaleFactor = (scaleFactor > 1.0) ? scaleFactor : 1.0;
    this->Update();
}


void vtkCudaRendererInformationHandler::SetMatrix(vtkMatrix4x4* matrix)
{
    //this->mat->DeepCopy(matrix);
    //this->mat->Invert();
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
        if (size[0] != this->RendererInfo.ActualResolution.x ||
            size[1] != this->RendererInfo.ActualResolution.y)
        {
            this->RendererInfo.ActualResolution.x = size[0];
            this->RendererInfo.ActualResolution.y = size[1];

            // HACK -> Allocate is too slow!!
            LocalZBuffer.Allocate<float>(this->RendererInfo.ActualResolution.x * this->RendererInfo.ActualResolution.y);
            CudaZBuffer.Allocate<float>(this->RendererInfo.ActualResolution.x * this->RendererInfo.ActualResolution.y);
        }
        this->RendererInfo.Resolution.x = this->RendererInfo.ActualResolution.x / this->RenderOutputScaleFactor;
        this->RendererInfo.Resolution.y = this->RendererInfo.ActualResolution.y / this->RenderOutputScaleFactor;

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
        this->RendererInfo.CameraDirection.x= cam->GetDirectionOfProjection()[0];
        this->RendererInfo.CameraDirection.y= cam->GetDirectionOfProjection()[1];
        this->RendererInfo.CameraDirection.z= cam->GetDirectionOfProjection()[2];
        this->RendererInfo.ViewUp.x = cam->GetViewUp()[0];
        this->RendererInfo.ViewUp.y = cam->GetViewUp()[1];
        this->RendererInfo.ViewUp.z = cam->GetViewUp()[2];

        float in[4];
        float out[4];

        in[3] = 0;

        unsigned int i;
        for (i = 0; i < 3; i++)
            in[i] = cam->GetPosition()[i];
        mat->MultiplyPoint(in, out);
        this->RendererInfo.CameraPos.x = out[0]; this->RendererInfo.CameraPos.y = out[1]; this->RendererInfo.CameraPos.z = out[2];

        for (i = 0; i < 3; i++)
            in[i] = cam->GetDirectionOfProjection()[i];
        mat->MultiplyPoint(in, out);
        this->RendererInfo.CameraDirection.x = out[0]; this->RendererInfo.CameraDirection.y = out[1]; this->RendererInfo.CameraDirection.z = out[2];

        for (i = 0; i < 3; i++)
            in[i] = cam->GetViewUp()[i];
        mat->MultiplyPoint(in, out);
        this->RendererInfo.ViewUp.x = out[0]; this->RendererInfo.ViewUp.y = out[1]; this->RendererInfo.ViewUp.z = out[2];


        float dot = this->RendererInfo.ViewUp.x * this->RendererInfo.CameraDirection.x +
              this->RendererInfo.ViewUp.y * this->RendererInfo.CameraDirection.y + 
              this->RendererInfo.ViewUp.z * this->RendererInfo.CameraDirection.z;

        this->RendererInfo.VerticalVec.x = this->RendererInfo.ViewUp.x - dot * this->RendererInfo.CameraDirection.x;
        this->RendererInfo.VerticalVec.y = this->RendererInfo.ViewUp.y - dot * this->RendererInfo.CameraDirection.y;
        this->RendererInfo.VerticalVec.z = this->RendererInfo.ViewUp.z - dot * this->RendererInfo.CameraDirection.z;

        this->RendererInfo.HorizontalVec.x = this->RendererInfo.VerticalVec.y * this->RendererInfo.CameraDirection.z - 
            this->RendererInfo.VerticalVec.z * this->RendererInfo.CameraDirection.y;
        this->RendererInfo.HorizontalVec.y = this->RendererInfo.VerticalVec.z * this->RendererInfo.CameraDirection.x - 
            this->RendererInfo.VerticalVec.x * this->RendererInfo.CameraDirection.z;
        this->RendererInfo.HorizontalVec.z = this->RendererInfo.VerticalVec.x * this->RendererInfo.CameraDirection.y - 
            this->RendererInfo.VerticalVec.y * this->RendererInfo.CameraDirection.x;


        double clipRange[2];
        cam->GetClippingRange(clipRange);
        this->RendererInfo.ClippingRange.x = (float)clipRange[0];
        this->RendererInfo.ClippingRange.y = (float)clipRange[1];


//        for (unsigned int i = 0 ; i < this->RendererInfo.Resolution.x * this->RendererInfo.Resolution.y; i++)
//            this->LocalZBuffer.GetMemPointerAs<float>()[i] = 100000;

        renWin->GetZbufferData(0,0,this->RendererInfo.Resolution.x-1, this->RendererInfo.Resolution.y-1, this->LocalZBuffer.GetMemPointerAs<float>());

        for (unsigned int i = 0; i < this->RendererInfo.Resolution.x * this->RendererInfo.Resolution.y; i++)
        {
         //   if (this->LocalZBuffer.GetMemPointerAs<float>()[i] != 1.0f)
            {
                float test  = this->LocalZBuffer.GetMemPointerAs<float>()[i] ;
               // cout << test<< "," << std::flush;
                float a = this->RendererInfo.ClippingRange.y / (this->RendererInfo.ClippingRange.y - this->RendererInfo.ClippingRange.x);
                float b = this->RendererInfo.ClippingRange.y * this->RendererInfo.ClippingRange.x / (this->RendererInfo.ClippingRange.x - this->RendererInfo.ClippingRange.y);
                float bla = b/(this->LocalZBuffer.GetMemPointerAs<float>()[i] - a);
                
               // cout << bla;

            }
        }
        this->LocalZBuffer.CopyTo(&this->CudaZBuffer);
        this->RendererInfo.ZBuffer = CudaZBuffer.GetMemPointerAs<float>();

    }
}
