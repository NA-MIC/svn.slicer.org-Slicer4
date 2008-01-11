#include "vtkVolumeCudaMapper.h"
#include "vtkVolumeRenderingCudaFactory.h"


#include "vtkActor.h"
#include "vtkPolyDataMapper.h"
#include "vtkPolyData.h"
#include "vtkTexture.h"
#include "vtkCellArray.h"
#include "vtkFloatArray.h"
#include "vtkPointData.h"
#include "vtkImageExtractComponents.h"

#include "vtkRenderer.h"
#include "vtkRenderWindow.h"
#include "vtkCamera.h"

#include "vtkImageData.h"
#include "vtkCudaMemory.h"
#include "vtkCudaHostMemory.h"

#include "vtkObjectFactory.h"

#include "vtkCudaMemory.h"
#include <vector_types.h>

extern "C" {
#include "CUDA_renderAlgo.h"
}



vtkCxxRevisionMacro(vtkVolumeCudaMapper, "$Revision: 1.6 $");
vtkStandardNewMacro(vtkVolumeCudaMapper);

vtkVolumeCudaMapper::vtkVolumeCudaMapper()
{
    this->LocalInputImage = NULL;
    this->LocalInputBuffer = vtkCudaLocalMemory::New();
    this->LocalOutputBuffer = vtkCudaHostMemory::New();
    this->CudaInputBuffer = vtkCudaMemory::New();
    this->CudaOutputBuffer = vtkCudaMemory::New();

    this->LocalOutputImage = vtkImageData::New();

    this->InitializeInternal();
    this->SetColor(255, 255, 255);
}  




vtkVolumeCudaMapper::~vtkVolumeCudaMapper()
{
    free(this->LocalInputBuffer);
    this->LocalOutputBuffer->Delete();
    this->CudaInputBuffer->Delete();
    this->CudaOutputBuffer->Delete();
    this->LocalOutputImage->Delete();
    this->LocalInputImage = NULL;
}


void vtkVolumeCudaMapper::InitializeInternal()
{
    unsigned int height = 128;
    unsigned int width = 128;


    //this->LocalInputBuffer->Allocate<unsigned char>(x*y*z);
    //memcpy(this->LocalInputBuffer->GetMemPointer(),
    //    data->GetScalarPointer(), sizeof(unsigned char) * x*y*z);
    //// HERE WE READ IN SOME DATA
    //try {
    //FILE *fp;
    //    fp=fopen("C:\\Documents and Settings\\bensch\\Desktop\\svn\\orxonox\\subprojects\\volrenSample\\heart256.raw","r");
    //    fread(this->LocalInputBuffer->GetMemPointer(), sizeof(unsigned char), x*y*z, fp);
    //    fclose(fp);}
    //catch (...)
    //{}

    //this->SetInput(this->LocalInputBuffer, x, y, z);


    //  cudaMallocHost( (void**) &h_renderAlgo_resultImage, sizeof(uchar4)*dsizeX*dsizeY));

    this->UpdateOutputResolution(width, height, 4);
}

void vtkVolumeCudaMapper::SetInput(vtkCudaLocalMemory* input,
                                   unsigned int x, unsigned int y, unsigned int z)
{
    this->LocalInputBuffer = input;

    this->InputDataSize[0] = x;
    this->InputDataSize[1] = y;
    this->InputDataSize[2] = z;

    //this->CudaInputBuffer->Allocate<uchar4>(x,y,z);
    this->LocalInputBuffer->CopyTo(this->CudaInputBuffer);
}

void vtkVolumeCudaMapper::SetInput(vtkImageData * input)
{
    this->LocalInputImage = input;

    if (input != NULL)
    {
        int* dims = input->GetDimensions();
        this->InputDataSize[0] = dims[0];
        this->InputDataSize[1] = dims[1];
        this->InputDataSize[2] = dims[2];

        this->CudaInputBuffer->CopyFrom(input);
    }
    else
    {

    }
}


void vtkVolumeCudaMapper::UpdateOutputResolution(unsigned int width, unsigned int height, unsigned int colors)
{
    // Set the data Size
    this->OutputDataSize[0] = width;
    this->OutputDataSize[1] = height;

    // Re-allocate the memory
    this->CudaOutputBuffer->Allocate<uchar4>(width * height);
    this->LocalOutputBuffer->Allocate<uchar4>(width * height);

    // Allocate the Image Data
    this->LocalOutputImage->SetScalarTypeToUnsignedChar();
    this->LocalOutputImage->SetNumberOfScalarComponents(4);
    this->LocalOutputImage->SetDimensions(width, height, 1);
    this->LocalOutputImage->SetExtent(0, width - 1, 
        0, height - 1, 
        0, 1 - 1);
    this->LocalOutputImage->SetNumberOfScalarComponents(colors);
    this->LocalOutputImage->SetScalarTypeToUnsignedChar();
    this->LocalOutputImage->AllocateScalars();
}

void vtkVolumeCudaMapper::Render(vtkRenderer *renderer, vtkVolume *volume)
{
    float color[6]={this->Color[0],this->Color[1],this->Color[2], 1,1,1};
    float minmax[6]={0,255,0,255,0,255};
    float lightVec[3]={0, 0, 1};


    vtkCamera* cam =
        renderer->GetActiveCamera();

    // Build the Rotation Matrix
    double ax,ay,az;
    double bx,by,bz;
    double cx,cy,cz;

    ax = cam->GetFocalPoint()[0] - cam->GetPosition()[0];
    ay = cam->GetFocalPoint()[1] - cam->GetPosition()[1];
    az = cam->GetFocalPoint()[2] - cam->GetPosition()[2];
    cam->GetViewUp(bx, by, bz);
    cx = ay*bz-az*by;
    cy = az*bx-ax*bz;
    cz = ax*by-ay*bx;

    bx = cy*az-cz*ay;
    by = cz*ax-cx*az;
    bz = cx*ay-cy*ax;

    double distance = cam->GetDistance();
    ax /= distance; ay /= distance; az /= distance;

    double len = sqrt(bx*bx + by*by + bz*bz);
    bx /= len; by /= len; bz /= len;

    len = sqrt(cx*cx + cy*cy + cz*cz);
    cx /= len; cy /= len; cz /= len;

    float rotationMatrix[4][4]=
    {{ax,bx,cx,0},
    {ay,by,cy,0},
    {az,bz,cz,0},
    {0,0,0,1}};
    /*  {{1,0,0,0},
    {0,1,0,0},
    {0,0,1,0},
    {0,0,0,1}};*/

    cerr << "Volume rendering.\n";
    // Do rendering. 

    CUDArenderAlgo_doRender(this->CudaOutputBuffer->GetMemPointerAs<uchar4>(),
        this->CudaInputBuffer->GetMemPointerAs<unsigned char>(),
        (float*)rotationMatrix, color, minmax, lightVec, 
        this->InputDataSize[0], this->InputDataSize[1], this->InputDataSize[2],    //3D data size
        this->OutputDataSize[0], this->OutputDataSize[1],     //result image size
        0,0,0,          //translation of data in x,y,z direction
        1, 1, 1,        //voxel dimension
        90, 255,        //min and max threshold
        -100);          //slicing distance from center of 3D data
    // Get the resulted image.

    cerr << "Copy result from GPU to CPU.\n";

    this->CudaOutputBuffer->CopyTo(this->LocalOutputImage);

    //memcpy(this->LocalOutputImage->GetScalarPointer(), this->LocalOutputBuffer->GetMemPointer(),
    //    sizeof(unsigned char) *
    //    this->OutputDataSize[0] *
    //    this->OutputDataSize[1] *
    //    4);


    vtkImageExtractComponents *components=vtkImageExtractComponents::New();
    components->SetInput(this->LocalOutputImage);
    components->SetComponents(0,1,2);


    vtkRenderWindow *renWin= renderer->GetRenderWindow();
    //Get current size of window
    int *size=renWin->GetSize();

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

    //Create the Polydata
    vtkPoints *points=vtkPoints::New();
    points->InsertPoint(0,coordinatesA);
    points->InsertPoint(1,coordinatesB);
    points->InsertPoint(2,coordinatesC);
    points->InsertPoint(3,coordinatesD);

    vtkCellArray *polygon=vtkCellArray::New();
    polygon->InsertNextCell(4);
    polygon->InsertCellPoint(0);
    polygon->InsertCellPoint(1);
    polygon->InsertCellPoint(2);
    polygon->InsertCellPoint(3);
    //Take care about Texture coordinates
    vtkFloatArray *textCoords=vtkFloatArray::New();
    textCoords->SetNumberOfComponents(2);
    textCoords->Allocate(8);
    float tc[2];
    tc[0]=0;
    tc[1]=0;
    textCoords->InsertNextTuple(tc);
    tc[0]=1;
    tc[1]=0;
    textCoords->InsertNextTuple(tc);
    tc[0]=1;
    tc[1]=1;
    textCoords->InsertNextTuple(tc);
    tc[0]=0;
    tc[1]=1;
    textCoords->InsertNextTuple(tc);

    vtkPolyData *polydata=vtkPolyData::New();
    polydata->SetPoints(points);
    polydata->SetPolys(polygon);
    polydata->GetPointData()->SetTCoords(textCoords);

    vtkPolyDataMapper *polyMapper=vtkPolyDataMapper::New();
    polyMapper->SetInput(polydata);

    vtkActor *actor=vtkActor::New(); 
    actor->SetMapper(polyMapper);

    //Take care about the texture
    vtkTexture *atext=vtkTexture::New();
    atext->SetInput(components->GetOutput());
    atext->SetInterpolate(1);
    actor->SetTexture(atext);

    //Remove all old Actors
    renderer->RemoveAllViewProps();

    renderer->AddActor(actor);
    //Remove the old Renderer

    renWin->SwapBuffersOn();


    //Delete everything we have done
    components->Delete();
    points->Delete();
    polygon->Delete();
    textCoords->Delete();
    polydata->Delete();
    polyMapper->Delete();
    actor->Delete();
    atext->Delete();
}

void vtkVolumeCudaMapper::PrintSelf(ostream& os, vtkIndent indent)
{
    vtkVolumeMapper::PrintSelf(os, indent);
}
