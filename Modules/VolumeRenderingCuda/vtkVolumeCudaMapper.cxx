#include "vtkVolumeCudaMapper.h"
#include "vtkVolumeRenderingCudaFactory.h"

#include <cutil.h>
#include <cuda_runtime_api.h>

#include "CUDA_renderAlgo.h"

vtkCxxRevisionMacro(vtkVolumeCudaMapper, "$Revision: 1.6 $");

//----------------------------------------------------------------------------
// Needed when we don't use the vtkStandardNewMacro.
vtkInstantiatorNewMacro(vtkVolumeCudaMapper);

vtkVolumeCudaMapper::vtkVolumeCudaMapper()
{
  this->inputBuffer= new char[256*256*256];

  FILE *fp;
  fp=fopen("heart256.raw","r");
  fread(this->inputBuffer, sizeof(unsigned char), 256*256*256, fp);
  fclose(fp);

  // Setting transformation matrix. This matrix will be used to do rotation and translation on ray tracing.

  float color[6]={255,255,255,1,1,1};
  float minmax[6]={0,255,0,255,0,255};
  float lightVec[3]={0, 0, 1};
  float rotationMatrix[4][4]={{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
  
  CUDArenderAlgo_init(256,256,256,1024,768);

  // Load 3D data into GPU memory.

  CUDArenderAlgo_loadData(this->inputBuffer, 256,256,256);
  
  // Do rendering. 
  CUDArenderAlgo_doRender((float*)rotationMatrix, color, minmax, lightVec, 
              256,256,256,    //3D data size
        1024,768,       //result image size
        0,0,0,          //translation of data in x,y,z direction
        1, 1, 1,        //voxel dimension
        90, 255,        //min and max threshold
        -100,           //slicing distance from center of 3D data
        outputBuffer);
  // Get the resulted image.
  CUDArenderAlgo_getResult((unsigned char**)&outputBuffer, 1024,768);
  
  fp=fopen("output.raw","w");
  fwrite(outputBuffer, sizeof(unsigned char), 1024*768*4, fp);
  fclose(fp);

  // Free allocated GPU memory.
}  


vtkVolumeCudaMapper::~vtkVolumeCudaMapper()
{
  CUDArenderAlgo_delete();
  free(inputBuffer);  
}


void vtkVolumeCudaMapper::PrintSelf(ostream& os, vtkIndent indent)
{
  vtkVolumeMapper::PrintSelf(os, indent);
}

vtkVolumeCudaMapper *vtkVolumeCudaMapper::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret =
      vtkVolumeRenderingCudaFactory::CreateInstance("vtkVolumeCUdaMapper");
  return (vtkVolumeCudaMapper*)ret;
}


int vtkVolumeCudaMapper::CheckSupportedCudaVersion(int cudaVersion)
{
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);

  int device;
  for (device = 0; device < deviceCount; ++device)
  {
    cudaDeviceProp deviceProperty;
    cudaGetDeviceProperties(&deviceProperty, device);
  }

  /// HACK
  return 0;
}

