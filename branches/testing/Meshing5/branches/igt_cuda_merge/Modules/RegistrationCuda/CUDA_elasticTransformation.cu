// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil.h>

// includes, kernels
#include "CUDA_elasticTransformation.h"

//#define USE_TIMER
#define WARP_SIZE 16
#define BLOCK_DIM3D 8  // this must be set to 8
#define BLOCK_DIM2D 16 // this must be set to 4 or more
#define ACC(X,Y,Z) ( ( (Z)*(sizeX)*(sizeY) ) + ( (Y)*(sizeX) ) + (X) )
#define SQR(X) ((X) * (X) )

__constant__ float c_elasticTransformation_inDataSize[3];
__constant__ float c_elasticTransformation_gridSize[3];
__constant__ float c_elasticTransformation_splineSize[3];

__device__ float getSplineValue(float x){
  float xabs=fabs(x);
  if(xabs<=1.0){
    return (1.0-((1-xabs/2.0)*x*x*1.5));
  }else if(xabs<2.0){
    return ((2-xabs)*(2-xabs)*(2-xabs)/4.0);
  }else{
    return 0;
  }
}
extern "C"
__global__ void CUDAkernel_elasticTransformation_doTransformation(unsigned char* inputData, unsigned char* outputData, float* transSpline, int tempPow)
{
  int xIndex = (blockDim.x*blockIdx.x + threadIdx.x) % (int)c_elasticTransformation_gridSize[0];
  int yIndex = (blockDim.x*blockIdx.x + threadIdx.x) / (int)c_elasticTransformation_gridSize[0];
  int zIndex = blockDim.y*blockIdx.y+ threadIdx.y;
  
  
  __shared__ float s_inDataSize[3];
  __shared__ float s_splineSize[3];
  __shared__ float s_newX[BLOCK_DIM2D*BLOCK_DIM2D];
  __shared__ float s_newY[BLOCK_DIM2D*BLOCK_DIM2D];
  __shared__ float s_newZ[BLOCK_DIM2D*BLOCK_DIM2D];
 
  __shared__ unsigned char s_outputData[BLOCK_DIM2D*BLOCK_DIM2D];
  
  int acc=threadIdx.y*BLOCK_DIM2D+threadIdx.x;

  if(acc<3){
    s_inDataSize[acc]=c_elasticTransformation_inDataSize[acc];
  }else if(acc<6){
    s_splineSize[acc-3]=c_elasticTransformation_splineSize[acc-3];
  }

  __syncthreads();

  int outIndex=zIndex*(int)s_inDataSize[0]*(int)s_inDataSize[1]+yIndex*(int)s_inDataSize[0]+xIndex;
  
  float relPosX, relPosY, relPosZ;
  float functemp;
  float tempValue;
  int a,b,c;
  
  s_newX[acc]=xIndex; s_newY[acc]=yIndex; s_newZ[acc]=zIndex;

  __syncthreads();

  if(xIndex<s_inDataSize[0] && yIndex < s_inDataSize[1] && zIndex < s_inDataSize[2]){

    relPosX=(float)(xIndex)*tempPow+tempPow;
    relPosY=(float)(yIndex)*tempPow+tempPow;
    relPosZ=(float)(zIndex)*tempPow+tempPow;
    for(a=(int)relPosX-1;a<(int)relPosX+3;a++){
      if(a<0||a>=s_splineSize[0])continue;
      for(b=(int)relPosY-1;b<(int)relPosY+3;b++){
          if(b<0||b>=s_splineSize[1])continue;
    for(c=(int)relPosZ-1;c<(int)relPosZ+3;c++){
      if(c<0||c>=s_splineSize[2])continue;
      
      if(a>=0 && a<=s_splineSize[0]-1 && b>=0 && b<=s_splineSize[1]-1 && c>=0 && c<=s_splineSize[2]-1 ){
        functemp=getSplineValue(relPosX-(float)a)*getSplineValue(relPosY-(float)b)*getSplineValue(relPosZ-(float)c);
        s_newX[acc]+=transSpline[c*(int)s_splineSize[1]*(int)s_splineSize[0]+b*(int)s_splineSize[0]+a]*functemp;
        s_newY[acc]+=transSpline[(int)s_splineSize[0]*(int)s_splineSize[1]*(int)s_splineSize[2]+c*(int)s_splineSize[1]*(int)s_splineSize[0]+b*(int)s_splineSize[0]+a]*functemp;
        s_newZ[acc]+=transSpline[2*(int)s_splineSize[0]*(int)s_splineSize[1]*(int)s_splineSize[2]+c*(int)s_splineSize[1]*(int)s_splineSize[0]+b*(int)s_splineSize[0]+a]*functemp;
      }else{
        functemp=0;
      }
      
      
    }
      }
    }
      

    if(s_newX[acc]>=0 && s_newX[acc]<=s_inDataSize[0]-1 && s_newY[acc]>=0 && s_newY[acc] <= s_inDataSize[1]-1 && s_newZ[acc]>=0 && s_newZ[acc] <= s_inDataSize[2]-1){
      tempValue=inputData[((int)s_newZ[acc])*(int)s_inDataSize[0]*(int)s_inDataSize[1]+((int)s_newY[acc])*(int)s_inDataSize[0]+((int)s_newX[acc])];
    }else{
      tempValue=0;
    }
    
    s_outputData[acc]=tempValue;

  }else{
    s_outputData[acc]=0;
  }
  
  if(xIndex<s_inDataSize[0] && yIndex < s_inDataSize[1] && zIndex < s_inDataSize[2]){
    outputData[outIndex]=s_outputData[acc];
  }

  return;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void CUDAelasticTransformation_doTransformation(unsigned char* inputData, unsigned char* outputData, float* transSpline, int inSizeX, int inSizeY, int inSizeZ, int splineSizeLevel)
{
  double tempPow;
  int tempSize; 
  
  if(splineSizeLevel==0){
    tempPow=1;
    tempSize=1;
  }else{
    tempPow=pow((float)2, splineSizeLevel);
    tempSize=(int)pow((float)2, -splineSizeLevel);
  }
  
  int splineSizeX=(int)(tempPow*(inSizeX+1)+1);
  int splineSizeY=(int)(tempPow*(inSizeY+1)+1);
  int splineSizeZ=(int)(tempPow*(inSizeZ+1)+1);
  
  // size of the matrix

  // setup execution parameters

  int gridSizeX, gridSizeY, gridSizeZ;

  if(inSizeX%BLOCK_DIM2D==0)gridSizeX=inSizeX;else gridSizeX=(inSizeX/BLOCK_DIM2D+1)*BLOCK_DIM2D;
  if(inSizeY%BLOCK_DIM2D==0)gridSizeY=inSizeY;else gridSizeY=(inSizeY/BLOCK_DIM2D+1)*BLOCK_DIM2D;
  if(inSizeZ%BLOCK_DIM2D==0)gridSizeZ=inSizeZ;else gridSizeZ=(inSizeZ/BLOCK_DIM2D+1)*BLOCK_DIM2D;

  dim3 grid(gridSizeX*gridSizeY / BLOCK_DIM2D, gridSizeZ / BLOCK_DIM2D, 1);
  dim3 threads(BLOCK_DIM2D, BLOCK_DIM2D, 1);

#ifdef USE_TIMER

  unsigned int timer;
  cutCreateTimer(&timer);

  cutStartTimer(timer);

#endif
  CUT_DEVICE_INIT();

  float inDataSize[3]={inSizeX, inSizeY, inSizeZ};
  CUDA_SAFE_CALL( cudaMemcpyToSymbol(c_elasticTransformation_inDataSize, inDataSize, sizeof(float)*3, 0));

  float splineSize[3]={splineSizeX, splineSizeY, splineSizeZ};
  CUDA_SAFE_CALL( cudaMemcpyToSymbol(c_elasticTransformation_splineSize, splineSize, sizeof(float)*3, 0));

  float gridSize[3]={gridSizeX, gridSizeY, gridSizeZ};
  CUDA_SAFE_CALL( cudaMemcpyToSymbol(c_elasticTransformation_gridSize, gridSize, sizeof(float)*3, 0));
  
#ifdef USE_TIMER
  cutStopTimer(timer);
  float naiveTime = cutGetTimerValue(timer);
  printf("Memory copy CPU to GPU average time:     %0.3f ms\n", naiveTime);fflush(stdout);
  cutResetTimer(timer);
  cutStartTimer(timer);
#endif

  CUDAkernel_elasticTransformation_doTransformation<<< grid, threads >>>(inputData, outputData, transSpline, tempPow);

  CUT_CHECK_ERROR("Kernel execution failed");

#ifdef USE_TIMER
  cutStopTimer(timer);
  naiveTime = cutGetTimerValue(timer);
  printf("Do rendering average time:     %0.3f ms\n", naiveTime);
#endif

  return;
}
