// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil.h>

// includes, kernels
#include "CUDA_linearTransformation.h"

//#define USE_TIMER
#define WARP_SIZE 16
#define BLOCK_DIM3D 8
#define BLOCK_DIM2D 16 // this must be set to 4 or more
#define ACC(X,Y,Z) ( ( (Z)*(sizeX)*(sizeY) ) + ( (Y)*(sizeX) ) + (X) )
#define SQR(X) ((X) * (X) )

__constant__ float c_linearTransformation_inDataSize[3];
__constant__ float c_linearTransformation_outDataSize[3];
__constant__ float c_linearTransformation_inDataThickness[3];
__constant__ float c_linearTransformation_outDataThickness[3];
__constant__ float c_linearTransformation_inverseMatrix[16];

__global__ void CUDAkernel_linearTransformation_doTransformation(unsigned char* d_linearTransformation_inputData, unsigned char* d_linearTransformation_outputData)
{
  int xIndex = blockDim.x*blockIdx.x + threadIdx.x;
  int yIndex = (blockDim.y*blockIdx.y)%(gridDim.y) + threadIdx.y;
  int zIndex = ((blockDim.y*blockIdx.y)/BLOCK_DIM3D/BLOCK_DIM3D) * blockDim.z + threadIdx.z;

  __shared__ float s_in[BLOCK_DIM3D*BLOCK_DIM3D*BLOCK_DIM3D*3];
  __shared__ float s_inverseMatrix[16];
  __shared__ float s_inDataSize[3];
  __shared__ float s_outDataSize[3];
  __shared__ float s_inDataThickness[3];
  __shared__ float s_outDataThickness[3];
  __shared__ unsigned char s_outputData[BLOCK_DIM3D*BLOCK_DIM3D*BLOCK_DIM3D];

  __shared__ float s_inOrigin[3];
  __shared__ float s_outOrigin[3];

  float vect_in[3];
  float vect_out[3];

  int acc=threadIdx.z*BLOCK_DIM3D*BLOCK_DIM3D+threadIdx.y*BLOCK_DIM3D+threadIdx.x;

  if(acc<16){
    s_inverseMatrix[acc]=c_linearTransformation_inverseMatrix[acc];
  }else if(acc>=16 && acc <19){
    s_inDataSize[acc-16]=c_linearTransformation_inDataSize[acc-16];
  }else if(acc>=19 && acc <22){
    s_outDataSize[acc-19]=c_linearTransformation_outDataSize[acc-19];
  }else if(acc>=22 && acc <25){
    s_inDataThickness[acc-22]=c_linearTransformation_inDataThickness[acc-22];
  }else if(acc>=25 && acc <28){
    s_outDataThickness[acc-25]=c_linearTransformation_outDataThickness[acc-25];
  }else if(acc>=28 && acc <31){
    s_inOrigin[acc-28]=c_linearTransformation_inDataSize[acc-28]*c_linearTransformation_inDataThickness[acc-28]/2.0;
  }else if(acc>=31 && acc <34){
    s_outOrigin[acc-31]=c_linearTransformation_outDataSize[acc-31]*c_linearTransformation_outDataThickness[acc-31]/2.0;;
  }

  __syncthreads();

  int outIndex=zIndex*(int)s_outDataSize[0]*(int)s_outDataSize[1]+yIndex*(int)s_outDataSize[0]+xIndex;

  vect_out[0]=((xIndex+0.5)*s_outDataThickness[0])-s_outOrigin[0];
  vect_out[1]=((yIndex+0.5)*s_outDataThickness[1])-s_outOrigin[1];
  vect_out[2]=((zIndex+0.5)*s_outDataThickness[2])-s_outOrigin[2];

  vect_in[0]=vect_out[0]*s_inverseMatrix[0]+vect_out[1]*s_inverseMatrix[1]+vect_out[2]*s_inverseMatrix[2]+s_inverseMatrix[3];
  vect_in[1]=vect_out[0]*s_inverseMatrix[4]+vect_out[1]*s_inverseMatrix[5]+vect_out[2]*s_inverseMatrix[6]+s_inverseMatrix[7];
  vect_in[2]=vect_out[0]*s_inverseMatrix[8]+vect_out[1]*s_inverseMatrix[9]+vect_out[2]*s_inverseMatrix[10]+s_inverseMatrix[11];

  s_in[acc]=(vect_in[0]+s_inOrigin[0])/s_inDataThickness[0]-0.5;
  s_in[acc+BLOCK_DIM3D*BLOCK_DIM3D*BLOCK_DIM3D]=(vect_in[1]+s_inOrigin[1])/s_inDataThickness[1]-0.5;
  s_in[acc+2*BLOCK_DIM3D*BLOCK_DIM3D*BLOCK_DIM3D]=(vect_in[2]+s_inOrigin[2])/s_inDataThickness[2]-0.5;

  int inX=(int)s_in[acc];
  int inY=(int)s_in[acc+BLOCK_DIM3D*BLOCK_DIM3D*BLOCK_DIM3D];
  int inZ=(int)s_in[acc+2*BLOCK_DIM3D*BLOCK_DIM3D*BLOCK_DIM3D];

  float decX= s_in[acc]-inX;
  float decY= s_in[acc+BLOCK_DIM3D*BLOCK_DIM3D*BLOCK_DIM3D]-inY;
  float decZ= s_in[acc+2*BLOCK_DIM3D*BLOCK_DIM3D*BLOCK_DIM3D]-inZ;

  __syncthreads();

  int temp;

  if(s_in[acc]>=0 && s_in[acc] < s_inDataSize[0]-2 && s_in[acc+BLOCK_DIM3D*BLOCK_DIM3D*BLOCK_DIM3D] >=0 && s_in[acc+BLOCK_DIM3D*BLOCK_DIM3D*BLOCK_DIM3D] < s_inDataSize[1]-2 && s_in[acc+2*BLOCK_DIM3D*BLOCK_DIM3D*BLOCK_DIM3D] >=0 && s_in[acc+2*BLOCK_DIM3D*BLOCK_DIM3D*BLOCK_DIM3D] < s_inDataSize[2]-2){
    temp=inZ*(int)s_inDataSize[0]*(int)s_inDataSize[1]+inY*(int)s_inDataSize[0]+inX;
    s_outputData[acc]=
      (1-decX)*(1-decY)*(1-decZ)*d_linearTransformation_inputData[temp]+
      (1-decX)*(1-decY)*(decZ)*d_linearTransformation_inputData[temp+(int)s_inDataSize[0]*(int)s_inDataSize[1]]+
      (1-decX)*(decY)*(1-decZ)*d_linearTransformation_inputData[temp+(int)s_inDataSize[0]]+
      (1-decX)*(decY)*(decZ)*d_linearTransformation_inputData[temp+(int)s_inDataSize[0]*(int)s_inDataSize[1]+(int)s_inDataSize[0]]+
      (decX)*(1-decY)*(1-decZ)*d_linearTransformation_inputData[temp+1]+
      (decX)*(1-decY)*(decZ)*d_linearTransformation_inputData[temp+(int)s_inDataSize[0]*(int)s_inDataSize[1]+1]+
      (decX)*(decY)*(1-decZ)*d_linearTransformation_inputData[temp+(int)s_inDataSize[0]+1]+
      (decX)*(decY)*(decZ)*d_linearTransformation_inputData[temp+(int)s_inDataSize[0]*(int)s_inDataSize[1]+(int)s_inDataSize[0]+1];
  }else{
    s_outputData[acc]=0;
  }
  d_linearTransformation_outputData[outIndex]=s_outputData[acc];
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
extern "C"
void CUDAlinearTransformation_doTransformation(unsigned char* inputData, unsigned char* outputData, float* inverseMatrix, int inSizeX, int inSizeY, int inSizeZ, int outSizeX, int outSizeY, int outSizeZ, float inThicknessX, float inThicknessY, float inThicknessZ, float outThicknessX, float outThicknessY, float outThicknessZ)
{
  // size of the matrix

  // setup execution parameters
 
  dim3 grid(outSizeX / BLOCK_DIM3D, outSizeY / BLOCK_DIM3D * outSizeZ /BLOCK_DIM3D, 1);
  dim3 threads(BLOCK_DIM3D, BLOCK_DIM3D, BLOCK_DIM3D);

#ifdef USE_TIMER

  unsigned int timer;
  cutCreateTimer(&timer);

  cutStartTimer(timer);

#endif
  CUT_DEVICE_INIT();

  float inDataSize[3]={inSizeX, inSizeY, inSizeZ};
  float inDataThickness[3]={inThicknessX, inThicknessY, inThicknessZ};
  float outDataThickness[3]={outThicknessX, outThicknessY, outThicknessZ};
  float outDataSize[3]={outSizeX, outSizeY, outSizeZ};

  CUDA_SAFE_CALL( cudaMemcpyToSymbol(c_linearTransformation_inDataSize, inDataSize, sizeof(float)*3, 0));
  CUDA_SAFE_CALL( cudaMemcpyToSymbol(c_linearTransformation_inDataThickness, inDataThickness, sizeof(float)*3, 0));
  CUDA_SAFE_CALL( cudaMemcpyToSymbol(c_linearTransformation_outDataThickness, outDataThickness, sizeof(float)*3, 0));
  CUDA_SAFE_CALL( cudaMemcpyToSymbol(c_linearTransformation_outDataSize, outDataSize, sizeof(float)*3, 0));

  CUDA_SAFE_CALL( cudaMemcpyToSymbol(c_linearTransformation_inverseMatrix, inverseMatrix, sizeof(float)*16, 0));

#ifdef USE_TIMER
  cutStopTimer(timer);
  float naiveTime = cutGetTimerValue(timer);
  printf("Memory copy CPU to GPU average time:     %0.3f ms\n", naiveTime);fflush(stdout);
  cutResetTimer(timer);
  cutStartTimer(timer);
#endif

  // execute the kernel
  CUDAkernel_linearTransformation_doTransformation<<< grid, threads >>>(inputData, outputData);

  CUT_CHECK_ERROR("Kernel execution failed");

#ifdef USE_TIMER
  cutStopTimer(timer);
  naiveTime = cutGetTimerValue(timer);
  printf("Do rendering average time:     %0.3f ms\n", naiveTime);
#endif

  return;
}
