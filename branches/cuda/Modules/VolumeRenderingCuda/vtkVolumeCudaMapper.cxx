// Type
#include "vtkVolumeCudaMapper.h"
#include "vtkVolumeRenderingCudaFactory.h"
#include "vtkObjectFactory.h"
// Extended Type
#include "vtkVolume.h"
#include "vtkPolyDataMapper.h"
#include "vtkVolumeProperty.h"
#include "vtkColorTransferFunction.h"
#include "vtkPiecewiseFunction.h"

//Data Types
#include "vtkPolyData.h"
#include "vtkTexture.h"
#include "vtkCellArray.h"
#include "vtkFloatArray.h"
#include "vtkPointData.h"
#include "vtkImageExtractComponents.h"

// Rendering
#include "vtkRenderer.h"
#include "vtkRenderWindow.h"
#include "vtkCamera.h"


// CUDA
#include "CudappSupport.h"
#include "vtkImageData.h"
#include "CudappDeviceMemory.h"
#include "CudappHostMemory.h"
#include "CudappMemoryArray.h"
#include "cudaRendererInformation.h"
#include "cudaVolumeInformation.h"

#include <vector_types.h>

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
    this->LocalOutputImage = vtkImageData::New();

    this->CudaInputBuffer = new Cudapp::DeviceMemory();
    this->CudaOutputBuffer = new Cudapp::DeviceMemory();

    this->Texture = 0;
    this->BufferObject = 0;

    this->SetThreshold(90,255);

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

    this->CudaColorTransferFunction = new Cudapp::DeviceMemory;
    this->CudaColorTransferFunction->Allocate<float3>(256);
    this->LocalColorTransferFunction = new Cudapp::HostMemory;
    this->LocalColorTransferFunction->Allocate<float3>(256);

    this->CudaAlphaTransferFunction = new Cudapp::DeviceMemory;
    this->CudaAlphaTransferFunction->Allocate<float>(256);
    this->LocalAlphaTransferFunction = new Cudapp::HostMemory;
    this->LocalAlphaTransferFunction->Allocate<float>(256);


    this->LocalZBuffer = new Cudapp::LocalMemory();
    this->CudaZBuffer = new Cudapp::DeviceMemory();
}  

vtkVolumeCudaMapper::~vtkVolumeCudaMapper()
{
    delete this->CudaInputBuffer;
    delete this->CudaOutputBuffer;
    this->LocalOutputImage->Delete();

    delete this->CudaColorTransferFunction;
    delete this->LocalColorTransferFunction;
    delete this->CudaAlphaTransferFunction;
    delete this->LocalAlphaTransferFunction;

    delete this->CudaZBuffer;

    if (this->Texture == 0 || !glIsTexture(this->Texture))
        glGenTextures(1, &this->Texture);

    if (this->GLBufferObjectsAvailiable == true)
        if (this->BufferObject != 0 && vtkgl::IsBufferARB(this->BufferObject))
            vtkgl::DeleteBuffersARB(1, &this->BufferObject);
}

void vtkVolumeCudaMapper::SetInput(vtkImageData * input)
{
    this->Superclass::SetInput(input);

    if (input != NULL)
    {
        this->CudaInputBuffer->AllocateBytes(input->GetActualMemorySize() * 1024);
        // We do this automatically
        this->CudaInputBuffer->CopyFrom(input->GetScalarPointer(), input->GetActualMemorySize() * 1024);
    }
    else
    {
        this->CudaInputBuffer->Free();
    }
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
    this->OutputDataSize[0] = width;
    this->OutputDataSize[1] = height;

    // Re-allocate the memory
    this->CudaOutputBuffer->Allocate<uchar4>(width * height);
    this->LocalZBuffer->Allocate<float>(width * height);
    this->CudaZBuffer->Allocate<float>(width * height);

    {
        // Allocate the Image Data
        this->LocalOutputImage->SetScalarTypeToUnsignedChar();
        this->LocalOutputImage->SetNumberOfScalarComponents(4);
        this->LocalOutputImage->SetDimensions(width, height, 1);
        this->LocalOutputImage->SetExtent(0, width - 1, 
            0, height - 1, 
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
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, this->LocalOutputImage->GetScalarPointer());
    glBindTexture(GL_TEXTURE_2D, 0);

    if (this->CurrentRenderMode == RenderToTexture)
    {
        // OpenGL Buffer Code
        if (this->BufferObject != 0 && vtkgl::IsBufferARB(this->BufferObject))
            vtkgl::DeleteBuffersARB(1, &this->BufferObject);
        vtkgl::GenBuffersARB(1, &this->BufferObject);
        vtkgl::BindBufferARB(vtkgl::PIXEL_UNPACK_BUFFER_ARB, this->BufferObject);
        vtkgl::BufferDataARB(vtkgl::PIXEL_UNPACK_BUFFER_ARB, width * height * 4, this->LocalOutputImage->GetScalarPointer(), vtkgl::STREAM_COPY);
        vtkgl::BindBufferARB(vtkgl::PIXEL_UNPACK_BUFFER_ARB, 0);
    }
}

void vtkVolumeCudaMapper::UpdateVolumeProperties(vtkVolumeProperty *property)
{
    //FILE *fp;
    //  unsigned char transferFunction[256*6];
    //
    //fp=fopen("C:\\color.map","r");
    //  fread(transferFunction, sizeof(unsigned char), 256*6, fp);
    //  fclose(fp);
    //
    //  float colorTransferFunction[256*3];
    //  float alphaTransferFunction[256];
    //  float zBuffer[1024*768];
    //
    //  int i;
    //  /*
    //  for(i=0;i<256;i++){
    //    colorTransferFunction[i*3]=i/255.0;
    //    colorTransferFunction[i*3+1]=0.7;
    //    colorTransferFunction[i*3+2]=(255-i)/255.0;
    //    alphaTransferFunction[i]=0.1;
    //  }
    //  */
    //
    //  for(i=0;i<256;i++){
    //    colorTransferFunction[i*3]=transferFunction[i*3]/255.0;
    //    colorTransferFunction[i*3+1]=transferFunction[i*3+1]/255.0;
    //    colorTransferFunction[i*3+2]=transferFunction[i*3+2]/255.0;
    //    alphaTransferFunction[i]=transferFunction[i+256*3]/255.0;
    //  }
    //  this->CudaColorTransferFunction->CopyFrom(colorTransferFunction, 256*3*sizeof(float));
    //  this->CudaAlphaTransferFunction->CopyFrom(alphaTransferFunction, 256 * sizeof(float));
    //

    double range[2];
    property->GetRGBTransferFunction()->GetRange(range);
    property->GetRGBTransferFunction()->GetTable(range[0], range[1], 256, this->LocalColorTransferFunction->GetMemPointerAs<float>());

    LocalColorTransferFunction->CopyTo(this->CudaColorTransferFunction);

    property->GetScalarOpacity()->GetTable(range[0], range[1], 256, this->LocalAlphaTransferFunction->GetMemPointerAs<float>());
    LocalAlphaTransferFunction->CopyTo(CudaAlphaTransferFunction);

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

    int width = size[0], height = size[1];
    this->UpdateOutputResolution(width, height);

    this->UpdateVolumeProperties(volume->GetProperty());

    vtkCamera* cam =
        renderer->GetActiveCamera();

    //for (unsigned int i = 0; i < 16; i++)
    //    rotationMatrix[i/4][i%4] = cam->GetPerspectiveTransformMatrix(1,0,1000)->GetElement(i/4,i%4);
    //

    int* dims = this->GetInput()->GetDimensions();

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
    cudaRendererInformation rendererInfo;
    rendererInfo.Resolution[0] = this->OutputDataSize[0];
    rendererInfo.Resolution[1] = this->OutputDataSize[1];

    std::vector<float3> lights;
    lights.push_back(make_float3(0,0,1));

    rendererInfo.LightCount = lights.size();
    if (!lights.empty())
        rendererInfo.LightVectors = &lights[0];

    rendererInfo.CameraPos[0] = cam->GetPosition()[0];
    rendererInfo.CameraPos[1] = cam->GetPosition()[1];
    rendererInfo.CameraPos[2] = cam->GetPosition()[2];
    rendererInfo.TargetPos[0] = cam->GetFocalPoint()[0];
    rendererInfo.TargetPos[1] = cam->GetFocalPoint()[1];
    rendererInfo.TargetPos[2] = cam->GetFocalPoint()[2];
    rendererInfo.ViewUp[0] = cam->GetViewUp()[0];
    rendererInfo.ViewUp[1] = cam->GetViewUp()[1];
    rendererInfo.ViewUp[2] = cam->GetViewUp()[2];
    rendererInfo.ZBuffer = this->CudaZBuffer->GetMemPointerAs<float>();
    rendererInfo.NearPlane = -500;
    rendererInfo.FarPlane = 1000;


    // Volume Information Setter.
    cudaVolumeInformation volumeInfo;
    volumeInfo.SourceData = this->CudaInputBuffer->GetMemPointer();
    volumeInfo.InputDataType = this->GetInput()->GetScalarType();

    volumeInfo.VoxelSize[0] = 1;
    volumeInfo.VoxelSize[1] = 1;
    volumeInfo.VoxelSize[2] = 1;
    volumeInfo.VolumeTransformation[0] =
        volumeInfo.VolumeTransformation[1] =
        volumeInfo.VolumeTransformation[2] = 0.0f;
    volumeInfo.VolumeSize[0] = dims[0];
    volumeInfo.VolumeSize[1] = dims[1];
    volumeInfo.VolumeSize[2] = dims[2];
    volumeInfo.MinThreshold = this->Threshold[0];
    volumeInfo.MaxThreshold = this->Threshold[1];
    volumeInfo.AlphaTransferFunction = this->CudaAlphaTransferFunction->GetMemPointerAs<float>();
    volumeInfo.ColorTransferFunction = this->CudaColorTransferFunction->GetMemPointerAs<float>();
    volumeInfo.FunctionSize = 256;
    volumeInfo.SteppingSize = 1.0;// nothing yet!!

    int* extent = this->GetInput()->GetExtent();
    volumeInfo.MinMaxValue[0] = volumeInfo.MinValueX = (float)extent[0];
    volumeInfo.MinMaxValue[1] = volumeInfo.MaxValueX = (float)extent[1];
    volumeInfo.MinMaxValue[2] = volumeInfo.MinValueY = (float)extent[2];
    volumeInfo.MinMaxValue[3] = volumeInfo.MaxValueY = (float)extent[3];
    volumeInfo.MinMaxValue[4] = volumeInfo.MinValueZ = (float)extent[4];
    volumeInfo.MinMaxValue[5] = volumeInfo.MaxValueZ = (float)extent[5];

    CUDArenderAlgo_doRender(RenderDestination,
        &rendererInfo,
        &volumeInfo);         

    //CUDArenderAlgo_doRender(RenderDestination,
    //    &rendererInfo,
    //    &volumeInfo
    //    );         

    // Get the resulted image.
    log->StopTimer();
    //vtkErrorMacro(<< "Elapsed Time to Render:: " << log->GetElapsedTime());

    log->StartTimer();
    if (this->CurrentRenderMode == RenderToTexture)
    {
        CUDA_SAFE_CALL(cudaGLUnmapBufferObject(this->BufferObject));
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, (0));
        CUDA_SAFE_CALL( cudaGLUnregisterBufferObject(this->BufferObject) );
        vtkgl::BindBufferARB(vtkgl::PIXEL_UNPACK_BUFFER_ARB, 0);
    }
    else // (this->CurrentRenderMode == RenderToMemory)
    {
        this->CudaOutputBuffer->CopyTo(this->LocalOutputImage->GetScalarPointer(), this->LocalOutputImage->GetActualMemorySize() * 1024);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, this->LocalOutputImage->GetScalarPointer());
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
