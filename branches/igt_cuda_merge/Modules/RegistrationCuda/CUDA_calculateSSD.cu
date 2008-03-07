// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil.h>

// includes, kernels
#include "CUDA_calculateSSD.h"

//#define USE_TIMER
#define WARP_SIZE 16
#define BLOCK_DIM2D 16 // this must be set to 16
#define ACC(X,Y,Z) ( ( (Z)*(sizeX)*(sizeY) ) + ( (Y)*(sizeX) ) + (X) )
#define SQR(X) ((X) * (X) )

__constant__ float c_calculateSSD_refDataSize[3];
__constant__ float c_calculateSSD_tarDataSize[3];
__constant__ float c_calculateSSD_refDataThickness[3];
__constant__ float c_calculateSSD_tarDataThickness[3];
__constant__ float c_calculateSSD_block[2];

__device__ float* SSDvalue;

__global__ void CUDAkernel_calculateSSD_doCalculation(unsigned char* d_calculateSSD_refData, unsigned char* d_calculateSSD_tarData, float* SSDvalue)
{
  int xIndex = (blockDim.x*blockIdx.x + threadIdx.x) % (int)c_calculateSSD_refDataSize[0];
  int yIndex = (blockDim.x*blockIdx.x + threadIdx.x) / (int)c_calculateSSD_refDataSize[0];
  int zIndex = blockDim.y*blockIdx.y+ threadIdx.y;
  

  int acc=threadIdx.y*BLOCK_DIM2D+threadIdx.x;

  __shared__ float s_refDataSize[3];
  __shared__ unsigned char s_refData[BLOCK_DIM2D*BLOCK_DIM2D];
  __shared__ unsigned char s_tarData[BLOCK_DIM2D*BLOCK_DIM2D];
  __shared__ float s_SSDvalue[BLOCK_DIM2D*BLOCK_DIM2D];

  if(acc<3){
    s_refDataSize[acc]=c_calculateSSD_refDataSize[acc];
  }

  s_SSDvalue[acc]=0;

  __syncthreads();

  int outIndex=zIndex*(int)s_refDataSize[0]*(int)s_refDataSize[1]+yIndex*(int)s_refDataSize[0]+xIndex;

  if(xIndex<s_refDataSize[0] && yIndex < s_refDataSize[1] && zIndex<s_refDataSize[2]){
    s_refData[acc]=d_calculateSSD_refData[outIndex];
    s_tarData[acc]=d_calculateSSD_tarData[outIndex];
  }else{
    s_refData[acc]=0;
    s_tarData[acc]=0;
  }
  
  if(xIndex<s_refDataSize[0] && yIndex < s_refDataSize[1] && zIndex<s_refDataSize[2]){
    s_SSDvalue[acc]=((int)s_refData[acc]-(int)s_tarData[acc])*((int)s_refData[acc]-(int)s_tarData[acc]);
  }else{
    s_SSDvalue[acc]=0;
  };

  __syncthreads();
  
  if(acc<BLOCK_DIM2D*BLOCK_DIM2D/2){
    s_SSDvalue[acc*2]+=s_SSDvalue[acc*2+1];
  }
  
  __syncthreads();

  if(acc<BLOCK_DIM2D*BLOCK_DIM2D/4){
    s_SSDvalue[acc*4]+=s_SSDvalue[acc*4+2];
  }
  
  __syncthreads();

  if(acc<BLOCK_DIM2D*BLOCK_DIM2D/8){
    s_SSDvalue[acc*8]+=s_SSDvalue[acc*8+4];
  }
  
  __syncthreads();

  if(acc<BLOCK_DIM2D*BLOCK_DIM2D/16){
    s_SSDvalue[acc*16]+=s_SSDvalue[acc*16+8];
  }
  
  __syncthreads();

  if(acc<BLOCK_DIM2D*BLOCK_DIM2D/32){
    s_SSDvalue[acc*32]+=s_SSDvalue[acc*32+16];
  }
  
  __syncthreads();
  
  if(acc<BLOCK_DIM2D*BLOCK_DIM2D/64){
    s_SSDvalue[acc*64]+=s_SSDvalue[acc*64+32];
  }
  
  __syncthreads();

  if(acc<BLOCK_DIM2D*BLOCK_DIM2D/128){
    s_SSDvalue[acc*128]+=s_SSDvalue[acc*128+64];
  }
  
  __syncthreads();

  if(acc<BLOCK_DIM2D*BLOCK_DIM2D/256){
    s_SSDvalue[acc*256]+=s_SSDvalue[acc*256+128];
  }
  
  __syncthreads();
  
  if(acc<1){
    if(blockIdx.x<c_calculateSSD_block[0] && blockIdx.y < c_calculateSSD_block[1]){
      SSDvalue[blockIdx.y*(int)c_calculateSSD_block[0]+blockIdx.x]=(float)s_SSDvalue[0];
      //SSDvalue[0]+=(float)s_SSDvalue[0];
    }
  }

  __syncthreads();
}

extern "C"
float CUDAcalculateSSD_doCalculation(unsigned char* refData, unsigned char* tarData, int refSizeX, int refSizeY, int refSizeZ){

  int blockX, blockY;

  blockX=((refSizeX*refSizeY-1)/BLOCK_DIM2D)+1;
  blockY=((refSizeZ-1)/BLOCK_DIM2D)+1;
  
  dim3 grid(blockX, blockY, 1);
  dim3 threads(BLOCK_DIM2D, BLOCK_DIM2D, 1);

#ifdef USE_TIMER

  unsigned int timer;
  cutCreateTimer(&timer);

  cutStartTimer(timer);

#endif
  printf("1\n");fflush(stdout);
  
  CUT_DEVICE_INIT();

  float refDataSize[3]={refSizeX, refSizeY, refSizeZ};
  CUDA_SAFE_CALL( cudaMemcpyToSymbol(c_calculateSSD_refDataSize, refDataSize, sizeof(float)*3, 0));
  
  float block[2]={blockX, blockY};
  CUDA_SAFE_CALL( cudaMemcpyToSymbol(c_calculateSSD_block, block, sizeof(float)*2, 0));

  CUDA_SAFE_CALL( cudaMalloc( (void**) &SSDvalue, sizeof(float)*blockX*blockY));

#ifdef USE_TIMER
  cutStopTimer(timer);
  float naiveTime = cutGetTimerValue(timer);
  printf("Memory copy CPU to GPU average time:     %0.3f ms\n", naiveTime);fflush(stdout);
  cutResetTimer(timer);
  cutStartTimer(timer);
#endif

  // execute the kernel
  CUDAkernel_calculateSSD_doCalculation<<< grid, threads >>>(refData, tarData, SSDvalue);

  CUT_CHECK_ERROR("Kernel execution failed");

  float *resultValue=(float*)malloc(blockX*blockY*sizeof(float));

  CUDA_SAFE_CALL( cudaMemcpy(resultValue, SSDvalue, blockX*blockY*sizeof(float),cudaMemcpyDeviceToHost));

  int i;
  float total=0;

  for(i=0;i<blockX*blockY;i++){
    total+=resultValue[i];
  }

  total/=(float)(refSizeX*refSizeY*refSizeZ);

  //printf("%lf \n", resultValue);

#ifdef USE_TIMER
  cutStopTimer(timer);
  naiveTime = cutGetTimerValue(timer);
  printf("SSD calculation average time:     %0.3f ms\n", naiveTime);
#endif

  CUDA_SAFE_CALL(cudaFree(SSDvalue));
  free(resultValue);

  return total;
}
