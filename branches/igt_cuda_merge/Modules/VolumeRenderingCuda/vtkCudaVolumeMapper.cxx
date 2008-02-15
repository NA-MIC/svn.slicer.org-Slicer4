// Type
#include "vtkCudaVolumeMapper.h"
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


vtkCxxRevisionMacro(vtkCudaVolumeMapper, "$Revision: 1.8 $");
vtkStandardNewMacro(vtkCudaVolumeMapper);

vtkCudaVolumeMapper::vtkCudaVolumeMapper()
{
    this->VolumeInfoHandler = vtkCudaVolumeInformationHandler::New();
    this->RendererInfoHandler = vtkCudaRendererInformationHandler::New();
}  

vtkCudaVolumeMapper::~vtkCudaVolumeMapper()
{
    this->VolumeInfoHandler->Delete();
    this->RendererInfoHandler->Delete();
}

void vtkCudaVolumeMapper::SetInput(vtkImageData * input)
{
    this->Superclass::SetInput(input);
    this->VolumeInfoHandler->SetInputData(input);
}

void vtkCudaVolumeMapper::SetRenderMode(int mode)
{
    //HACK
    //this->MemoryTexture->SetRenderMode(mode);
}

int vtkCudaVolumeMapper::GetCurrentRenderMode() const
{
    //HACK
    return 0; //this->MemoryTexture->GetCurrentRenderMode();
    //TODO
}

/**
 * @brief sets the Threshold of the Input Array
 */
void vtkCudaVolumeMapper::SetThreshold(unsigned int min, unsigned int max)
{
    this->VolumeInfoHandler->SetThreshold(min, max);
}

void vtkCudaVolumeMapper::SetSteppingSize(float steppingSize)
{
    this->VolumeInfoHandler->SetSteppingSize(steppingSize);
}


#include "vtkTimerLog.h"

void vtkCudaVolumeMapper::Render(vtkRenderer *renderer, vtkVolume *volume)
{
    // This should update the the CudaInputBuffer only when needed.
    //if (this->GetInput()->GetMTime() > this->GetMTime())
    //  this->CudaInputBuffer->CopyFrom(this->GetInput()->GetScalarPointer(), this->GetInput()->GetActualMemorySize() * 1024);

    vtkRenderWindow *renWin= renderer->GetRenderWindow();
    //Get current size of window
    int *size=renWin->GetSize();
    //int width = size[0], height = size[1];

    vtkTimerLog* log = vtkTimerLog::New();
    log->StartTimer();
    // Renderer Information Setter.
    this->VolumeInfoHandler->SetInputData(this->GetInput());
    this->VolumeInfoHandler->SetVolume(volume);
    this->VolumeInfoHandler->Update();

    this->RendererInfoHandler->SetRenderer(renderer);
    this->RendererInfoHandler->Bind();

    CUDArenderAlgo_doRender(
        this->RendererInfoHandler->GetRendererInfo(),
        this->VolumeInfoHandler->GetVolumeInfo());         

    log->StopTimer();
    //vtkErrorMacro(<< "Elapsed Time to Render:: " << log->GetElapsedTime());

//    renderer->SetBackground(this->renViewport->GetBackground());
//    renderer->SetActiveCamera(this->renViewport->GetActiveCamera());

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
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

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
    this->RendererInfoHandler->Unbind();

    log->Delete();
    return;
}

void vtkCudaVolumeMapper::PrintSelf(ostream& os, vtkIndent indent)
{
    vtkVolumeMapper::PrintSelf(os, indent);
}
