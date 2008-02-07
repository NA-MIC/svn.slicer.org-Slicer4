// Type
#include "vtkVolumeCudaMapper.h"
#include "vtkVolumeRenderingCudaFactory.h"
#include "vtkObjectFactory.h"
// Extended Type
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
#include "CudappMemoryArray.h"
#include "cudaRendererInformation.h"
#include "cudaVolumeInformation.h"
#include <vector_types.h>
// VTKCUDA
#include "vtkCudaVolumeInformationHandler.h"
#include "vtkCudaRendererInformationHandler.h"

// openGL
#include "vtkOpenGLExtensionManager.h"
#include "vtkgl.h"
#include "cuda_gl_interop.h"

extern "C" {
#include "CUDA_renderAlgo.h"
}
#include <cutil.h>


vtkCxxRevisionMacro(vtkVolumeCudaMapper, "$Revision: 1.6 $");

vtkVolumeCudaMapper* vtkVolumeCudaMapper::New()
{
    Cudapp::Support* support = new Cudapp::Support();
    bool cudaIsSupported = support->IsSupported();
    delete support;
    if (cudaIsSupported)
        return new vtkVolumeCudaMapper;
    else
        return new vtkVolumeCudaMapper; // HACK

}

vtkVolumeCudaMapper::vtkVolumeCudaMapper()
{
    this->VolumeInfoHandler = vtkCudaVolumeInformationHandler::New();
    this->RendererInfoHandler = vtkCudaRendererInformationHandler::New();

    this->LocalOutputImage = vtkImageData::New();

    this->CudaOutputBuffer = new Cudapp::DeviceMemory();

    this->Texture = 0;
    this->BufferObject = 0;

    this->OutputDataSize[0] = this->OutputDataSize[1] = 0;

    // check for the RenderMode
    vtkOpenGLExtensionManager *extensions = vtkOpenGLExtensionManager::New();
    extensions->SetRenderWindow(NULL);
    if (extensions->ExtensionSupported("GL_ARB_vertex_buffer_object"))
    {
        extensions->LoadExtension("GL_ARB_vertex_buffer_object");
        this->GLBufferObjectsAvailiable = true;
        this->CurrentRenderMode = RenderToTexture;
    }
    else
    {
        this->GLBufferObjectsAvailiable = false;
        this->CurrentRenderMode = RenderToMemory;
    }
    extensions->Delete();

    this->LocalZBuffer = new Cudapp::LocalMemory();
    this->CudaZBuffer = new Cudapp::DeviceMemory();
}  

vtkVolumeCudaMapper::~vtkVolumeCudaMapper()
{
    delete this->CudaOutputBuffer;
    this->LocalOutputImage->Delete();

    delete this->CudaZBuffer;

    if (this->Texture == 0 || !glIsTexture(this->Texture))
        glGenTextures(1, &this->Texture);

    if (this->GLBufferObjectsAvailiable == true)
        if (this->BufferObject != 0 && vtkgl::IsBufferARB(this->BufferObject))
            vtkgl::DeleteBuffersARB(1, &this->BufferObject);

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
    if (mode == RenderToTexture && this->GLBufferObjectsAvailiable)
    {
        this->CurrentRenderMode = mode;
    }
    else
    {
        this->CurrentRenderMode = RenderToMemory;
    }
    this->UpdateOutputResolution(this->OutputDataSize[0], this->OutputDataSize[1], true);
}

void vtkVolumeCudaMapper::UpdateOutputResolution(unsigned int width, unsigned int height, bool TypeChanged)
{
    if (this->OutputDataSize[0] == width &&
        this->OutputDataSize[1] == height && !TypeChanged)
        return;

    // Set the data Size
    this->OutputDataSize[0] = width ;
    this->OutputDataSize[1] = height;

    // Re-allocate the memory
    this->CudaOutputBuffer->Allocate<uchar4>(this->OutputDataSize[0] * this->OutputDataSize[1]);
    this->LocalZBuffer->Allocate<float>(this->OutputDataSize[0] * this->OutputDataSize[1]);
    this->CudaZBuffer->Allocate<float>(this->OutputDataSize[0] * this->OutputDataSize[1]);

    {
        // Allocate the Image Data
        this->LocalOutputImage->SetScalarTypeToUnsignedChar();
        this->LocalOutputImage->SetNumberOfScalarComponents(4);
        this->LocalOutputImage->SetDimensions(this->OutputDataSize[0], this->OutputDataSize[1], 1);
        this->LocalOutputImage->SetExtent(0, this->OutputDataSize[0] - 1, 
            0, this->OutputDataSize[1] - 1, 
            0, 1 - 1);
        this->LocalOutputImage->SetNumberOfScalarComponents(4);
        this->LocalOutputImage->AllocateScalars();
    }
    // TEXTURE CODE
    glEnable(GL_TEXTURE_2D);
    if (this->Texture != 0 && glIsTexture(this->Texture))
        glDeleteTextures(1, &this->Texture);
    glGenTextures(1, &this->Texture);
    glBindTexture(GL_TEXTURE_2D, this->Texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, this->OutputDataSize[0], this->OutputDataSize[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, this->LocalOutputImage->GetScalarPointer());
    glBindTexture(GL_TEXTURE_2D, 0);

    if (this->CurrentRenderMode == RenderToTexture)
    {
        // OpenGL Buffer Code
        if (this->BufferObject != 0 && vtkgl::IsBufferARB(this->BufferObject))
            vtkgl::DeleteBuffersARB(1, &this->BufferObject);
        vtkgl::GenBuffersARB(1, &this->BufferObject);
        vtkgl::BindBufferARB(vtkgl::PIXEL_UNPACK_BUFFER_ARB, this->BufferObject);
        vtkgl::BufferDataARB(vtkgl::PIXEL_UNPACK_BUFFER_ARB, this->OutputDataSize[0] * this->OutputDataSize[1] * 4, this->LocalOutputImage->GetScalarPointer(), vtkgl::STREAM_COPY);
        vtkgl::BindBufferARB(vtkgl::PIXEL_UNPACK_BUFFER_ARB, 0);
    }
}

#include "vtkTimerLog.h"
#include "texture_types.h"
#include "vtkMatrix4x4.h"
#include "vector_functions.h"

void vtkVolumeCudaMapper::Render(vtkRenderer *renderer, vtkVolume *volume)
{
    // Renice!!
    int* minMax = this->GetInput()->GetExtent();
    float minmax[6]; //={minMax[0],minMax[1],minMax[2],minMax[3],minMax[4],minMax[5]};
    for (unsigned int i = 0; i < 6; i++)
        minmax[i] = minMax[i];
    float lightVec[3]={0, 0, 1};

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

    this->VolumeInfoHandler->SetInputData(this->GetInput());
    this->VolumeInfoHandler->SetVolume(volume);
    this->VolumeInfoHandler->Update();

    // Do rendering.
    uchar4* RenderDestination = NULL;
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, this->Texture);

    if (this->CurrentRenderMode == RenderToTexture)
    {
        CUDA_SAFE_CALL( cudaGLRegisterBufferObject(this->BufferObject) );

        vtkgl::BindBufferARB(vtkgl::PIXEL_UNPACK_BUFFER_ARB, this->BufferObject);
        CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&RenderDestination, this->BufferObject));
    }
    else
    {
        RenderDestination = this->CudaOutputBuffer->GetMemPointerAs<uchar4>();
    }
    vtkTimerLog* log = vtkTimerLog::New();
    log->StartTimer();


    // Renderer Information Setter.
    this->RendererInfoHandler->SetRenderer(renderer);
    this->RendererInfoHandler->SetZBuffer(this->CudaZBuffer);

    CUDArenderAlgo_doRender(RenderDestination,
        this->RendererInfoHandler->GetRendererInfo(),
        this->VolumeInfoHandler->GetVolumeInfo());         

    // Get the resulted image.
    log->StopTimer();
    //vtkErrorMacro(<< "Elapsed Time to Render:: " << log->GetElapsedTime());

    log->StartTimer();
    if (this->CurrentRenderMode == RenderToTexture)
    {
        CUDA_SAFE_CALL(cudaGLUnmapBufferObject(this->BufferObject));
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, this->OutputDataSize[0], this->OutputDataSize[1], GL_RGBA, GL_UNSIGNED_BYTE, (0));
        CUDA_SAFE_CALL( cudaGLUnregisterBufferObject(this->BufferObject) );
        vtkgl::BindBufferARB(vtkgl::PIXEL_UNPACK_BUFFER_ARB, 0);
    }
    else // (this->CurrentRenderMode == RenderToMemory)
    {
        this->CudaOutputBuffer->CopyTo(this->LocalOutputImage->GetScalarPointer(), this->LocalOutputImage->GetActualMemorySize() * 1024);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, this->OutputDataSize[0], this->OutputDataSize[1], GL_RGBA, GL_UNSIGNED_BYTE, this->LocalOutputImage->GetScalarPointer());
    }
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

    log->Delete();

    return;
}

void vtkVolumeCudaMapper::PrintSelf(ostream& os, vtkIndent indent)
{
    vtkVolumeMapper::PrintSelf(os, indent);
}
