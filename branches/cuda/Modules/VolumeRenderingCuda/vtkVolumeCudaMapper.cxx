// Type
#include "vtkVolumeCudaMapper.h"
#include "vtkVolumeRenderingCudaFactory.h"
#include "vtkObjectFactory.h"

// Volume
#include "vtkVolume.h"
#include "vtkVolumeProperty.h"

// Rendering
#include "vtkCamera.h"
#include "vtkRenderer.h"
#include "vtkRenderWindow.h"

// CUDA
#include "CudappSupport.h"
#include "vtkImageData.h"
#include "CudappDeviceMemory.h"
#include "CudappHostMemory.h"

// VTKCUDA
#include "vtkCudaVolumeInformationHandler.h"
#include "vtkCudaRendererInformationHandler.h"
#include "vtkCudaMemoryTexture.h"

#include "vtkgl.h"
extern "C" {
#include "CUDA_renderAlgo.h"
}


vtkCxxRevisionMacro(vtkVolumeCudaMapper, "$Revision: 1.8 $");
vtkStandardNewMacro(vtkVolumeCudaMapper);

vtkVolumeCudaMapper::vtkVolumeCudaMapper()
{
    this->VolumeInfoHandler = vtkCudaVolumeInformationHandler::New();
    this->RendererInfoHandler = vtkCudaRendererInformationHandler::New();
    this->MemoryTexture = vtkCudaMemoryTexture::New();

    this->LocalOutputImage = vtkImageData::New();

    this->OutputDataSize[0] = this->OutputDataSize[1] = 0;

    this->LocalZBuffer = new Cudapp::LocalMemory();
    this->CudaZBuffer = new Cudapp::DeviceMemory();
}  

vtkVolumeCudaMapper::~vtkVolumeCudaMapper()
{
    this->LocalOutputImage->Delete();

    delete this->CudaZBuffer;

    this->MemoryTexture->Delete();
    this->VolumeInfoHandler->Delete();
    this->RendererInfoHandler->Delete();
}

void vtkVolumeCudaMapper::SetInput(vtkImageData * input)
{
    this->Superclass::SetInput(input);
    this->VolumeInfoHandler->SetInputData(input);
}

void vtkVolumeCudaMapper::SetRenderMode(vtkVolumeCudaMapper::RenderMode mode)
{
    this->MemoryTexture->SetRenderMode(mode);
}

int vtkVolumeCudaMapper::GetCurrentRenderMode() const
{
    return this->MemoryTexture->GetCurrentRenderMode();
    //TODO
}

/**
 * @brief sets the Threshold of the Input Array
 */
void vtkVolumeCudaMapper::SetThreshold(unsigned int min, unsigned int max)
{
    this->VolumeInfoHandler->SetThreshold(min, max);
}

void vtkVolumeCudaMapper::UpdateOutputResolution(unsigned int width, unsigned int height, bool TypeChanged)
{
    if (this->OutputDataSize[0] == width &&
        this->OutputDataSize[1] == height && !TypeChanged)
        return;
    // Set the data Size
    this->OutputDataSize[0] = width;
    this->OutputDataSize[1] = height;

    // Re-allocate the memory
    this->LocalZBuffer->Allocate<float>(this->OutputDataSize[0] * this->OutputDataSize[1]);
    this->CudaZBuffer->Allocate<float>(this->OutputDataSize[0] * this->OutputDataSize[1]);

    this->MemoryTexture->SetSize(width, height);
}

#include "vtkTimerLog.h"

void vtkVolumeCudaMapper::Render(vtkRenderer *renderer, vtkVolume *volume)
{
    for (unsigned int i = 0 ; i < (this->OutputDataSize[0]) * this->OutputDataSize[1]; i++)
        this->LocalZBuffer->GetMemPointerAs<float>()[i] = 100000;
    //renderer->GetRenderWindow()->GetZbufferData(0,0,this->OutputDataSize[0]-1, this->OutputDataSize[1]-1, this->LocalZBuffer->GetMemPointerAs<float>());
    this->LocalZBuffer->CopyTo(this->CudaZBuffer);


    // This should update the the CudaInputBuffer only when needed.
    //if (this->GetInput()->GetMTime() > this->GetMTime())
    //  this->CudaInputBuffer->CopyFrom(this->GetInput()->GetScalarPointer(), this->GetInput()->GetActualMemorySize() * 1024);

    vtkRenderWindow *renWin= renderer->GetRenderWindow();
    //Get current size of window
    int *size=renWin->GetSize();
    //int width = size[0], height = size[1];
    this->UpdateOutputResolution(size[0], size[1]);

    // Do rendering.
    this->MemoryTexture->BindTexture();
    this->MemoryTexture->BindBuffer();

    vtkTimerLog* log = vtkTimerLog::New();
    log->StartTimer();

    // Renderer Information Setter.
    this->RendererInfoHandler->SetRenderer(renderer);
    this->RendererInfoHandler->SetZBuffer(this->CudaZBuffer);

    this->VolumeInfoHandler->SetInputData(this->GetInput());
    this->VolumeInfoHandler->SetVolume(volume);
    this->VolumeInfoHandler->Update();

    CUDArenderAlgo_doRender((uchar4*)this->MemoryTexture->GetRenderDestination(),
        this->RendererInfoHandler->GetRendererInfo(),
        this->VolumeInfoHandler->GetVolumeInfo());         

    // Get the resulted image.
    log->StopTimer();
    //vtkErrorMacro(<< "Elapsed Time to Render:: " << log->GetElapsedTime());
    log->StartTimer();
    this->MemoryTexture->UnbindBuffer();

    log->StopTimer();
    //vtkErrorMacro(<< "Elapsed Time to Copy Memory:: " << log->GetElapsedTime());

    log->StartTimer();

    //renderer->SetBackground(this->renViewport->GetBackground());
    //renderer->SetActiveCamera(this->renViewport->GetActiveCamera());

    renderer->SetDisplayPoint(0,0,0.5);
    renderer->DisplayToWorld();
    double coordinatesA[4];
    renderer->GetWorldPoint(coordinatesA);

    renderer->SetDisplayPoint(size[0],0,0.5);
    renderer->DisplayToWorld();
    double coordinatesB[4];
    renderer->GetWorldPoint(coordinatesB);

    renderer->SetDisplayPoint(size[0],size[1],0.5);
    renderer->DisplayToWorld();
    double coordinatesC[4];
    renderer->GetWorldPoint(coordinatesC);

    renderer->SetDisplayPoint(0,size[1],0.5);
    renderer->DisplayToWorld();
    double coordinatesD[4];
    renderer->GetWorldPoint(coordinatesD);


    glPushAttrib(GL_BLEND);
    glEnable(GL_BLEND);
    glPushAttrib(GL_LIGHTING);
    glDisable(GL_LIGHTING);

    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    glBegin(GL_QUADS);
    glTexCoord2i(1,0);
    glVertex4dv(coordinatesA);
    glTexCoord2i(0,0);
    glVertex4dv(coordinatesB);
    glTexCoord2i(0,1);
    glVertex4dv(coordinatesC);
    glTexCoord2i(1,1);
    glVertex4dv(coordinatesD);
    glEnd();
    glPopAttrib();
    glPopAttrib();
    this->MemoryTexture->UnbindTexture();

    log->Delete();
    return;
}

void vtkVolumeCudaMapper::PrintSelf(ostream& os, vtkIndent indent)
{
    vtkVolumeMapper::PrintSelf(os, indent);
}
