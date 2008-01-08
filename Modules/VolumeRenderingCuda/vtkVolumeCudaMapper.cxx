#include "vtkVolumeCudaMapper.h"
#include "vtkVolumeRenderingCudaFactory.h"

#include <cuda_runtime_api.h>

extern "C" {
#include "CUDA_renderAlgo.h"
}

#include "vtkActor.h"
#include "vtkPolyDataMapper.h"
#include "vtkPolyData.h"
#include "vtkTexture.h"
#include "vtkCellArray.h"
#include "vtkFloatArray.h"
#include "vtkPointData.h"
#include "vtkDataSetReader.h"
#include "vtkImageReader.h"
#include "vtkPNGReader.h"
#include "vtkImageViewer.h"
#include "vtkObjectFactory.h"


vtkCxxRevisionMacro(vtkVolumeCudaMapper, "$Revision: 1.6 $");
vtkStandardNewMacro(vtkVolumeCudaMapper);

vtkVolumeCudaMapper::vtkVolumeCudaMapper()
{
  this->inputBuffer= (unsigned char*)malloc(256*256*256*sizeof(unsigned char));

  FILE *fp;
  fp=fopen("/projects/igtdev/bensch/svn/volrenSample/heart256.raw","r");
  fread(this->inputBuffer, sizeof(unsigned char), 256*256*256, fp);
  fread(this->inputBuffer, sizeof(unsigned char), 256*256*256, fp);
  fread(this->inputBuffer, sizeof(unsigned char), 256*256*256, fp);
  fclose(fp);
  
  printf("FINISHED CREATING WINDOW\n");
  CUDArenderAlgo_init(256,256,256,1024,768);

  // Load 3D data into GPU memory.

  CUDArenderAlgo_loadData(this->inputBuffer, 256,256,256);
}  


vtkVolumeCudaMapper::~vtkVolumeCudaMapper()
{
  // Free allocated GPU memory.
  CUDArenderAlgo_delete();
  free(inputBuffer);  
}


void vtkVolumeCudaMapper::Render(vtkRenderer *renderer, vtkVolume *volume)
{
   // Setting transformation matrix. This matrix will be used to do rotation and translation on ray tracing.
  float color[6]={255,255,255,1,1,1};
  float minmax[6]={0,255,0,255,0,255};
  float lightVec[3]={0, 0, 1};
  float rotationMatrix[4][4]={{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
  
  /*
    // Do rendering. 
  CUDArenderAlgo_doRender((float*)rotationMatrix, color, minmax, lightVec, 
        256,256,256,    //3D data size
        1024,768,       //result image size
        0,0,0,          //translation of data in x,y,z direction
        1, 1, 1,        //voxel dimension
        90, 255,        //min and max threshold
        -100           //slicing distance from center of 3D data
        );
  // Get the resulted image.
  CUDArenderAlgo_getResult((unsigned char**)&outputBuffer, 1024,768);
  
  FILE* fp=fopen("output.raw","w");
  fwrite(outputBuffer, sizeof(unsigned char), 1024*768*4, fp);
  fclose(fp);
  */
  
  /*
  outputBuffer = (unsigned char*) malloc(1024 * 786 * 4 * sizeof(unsigned char));
  FILE* fp = fopen ("output.raw", "r");
  fread(this->outputBuffer, sizeof(unsigned char), 1024 * 786 * 4, fp);
  fclose(fp);
  */
  
  
  //printf ("Testing the CUDA implementation here\n");
  
};

void vtkVolumeCudaMapper::PrintSelf(ostream& os, vtkIndent indent)
{
  vtkVolumeMapper::PrintSelf(os, indent);
}

void vtkVolumeCudaMapper::PrepareRender()
{
   double coordinatesA[4] = {0,0,0,0};
   double coordinatesB[4] = {1,0,0,0};
   double coordinatesC[4] = {1,1,0,0};
   double coordinatesD[4] = {0,1,0,0};
  
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
        
        
//        atext->SetInput(components->GetOutput());
        atext->SetInterpolate(1);
        actor->SetTexture(atext);

/*
        //Remove all old Actors
        this->renPlane->RemoveAllViewProps();

        this->renPlane->AddActor(actor);
        //Remove the old Renderer

        renWin->SwapBuffersOn();
        //Do we do that
*/



        //Delete everything we have done
//        image->Delete();
 //       imageData->Delete();
//        components->Delete();
        points->Delete();
        polygon->Delete();
        textCoords->Delete();
        polydata->Delete();
        polyMapper->Delete();
        actor->Delete();
        atext->Delete();
       // this->Gui->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->GetRenderWindow()->Render();  
}
