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
    //  this->CudaInputBuffer->CopyFrom(this->GetInput()->GetScalarPointer(), this->GetInput()->GetScalarSize());

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

    // Enter 2D Mode
    glPushAttrib(GL_ENABLE_BIT);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0.0, 1.0, 1.0, 0.0, 0.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    // Actual Rendering
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    glBegin(GL_QUADS);
    glTexCoord2i(1,0);   glVertex2i(0,1);
    glTexCoord2i(0,0);   glVertex2i(1,1);
    glTexCoord2i(0,1);   glVertex2i(1,0);
    glTexCoord2i(1,1);   glVertex2i(0,0);
    glEnd();
    this->RendererInfoHandler->Unbind();

    // Leave the 2D Mode again.
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glPopAttrib();

    log->Delete();
    return;
}

void vtkCudaVolumeMapper::PrintSelf(ostream& os, vtkIndent indent)
{
    vtkVolumeMapper::PrintSelf(os, indent);
}
