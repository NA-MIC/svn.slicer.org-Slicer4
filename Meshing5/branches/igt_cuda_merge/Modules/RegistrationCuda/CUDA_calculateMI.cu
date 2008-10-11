// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil.h>

// includes, kernels
#include "CUDA_calculateMI.h"

//#define USE_TIMER
#define WARP_SIZE 16
#define BLOCK_DIM3D 8
#define BLOCK_DIM2D 4 // this must be set to 2 or more
#define BLOCK_DIM1D 256
#define ACC(X,Y,Z) ( ( (Z)*(sizeX)*(sizeY) ) + ( (Y)*(sizeX) ) + (X) )
#define SQR(X) ((X) * (X) )

__constant__ float c_calculateMI_refDataSize[3];
__constant__ float c_calculateMI_tarDataSize[3];
__constant__ float c_calculateMI_refDataThickness[3];
__constant__ float c_calculateMI_tarDataThickness[3];

__device__ float* d_calculateMI_histogram;
__device__ float* d_calculateMI_value;

float* h_calculateMI_histogram;
float* h_calculateMI_value;

__global__ void CUDAkernel_calculateMI_initHistogram(float* d_calculateMI_histogram)
{
  int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
  
  int acc = yIndex*256+xIndex;
  int i;
  
  for(i=0;i<BLOCK_DIM2D*BLOCK_DIM2D;i++){
    d_calculateMI_histogram[acc+i*256*256]=0.0;
  }
}


__global__ void CUDAkernel_calculateMI_fillHistogram(unsigned char* d_calculateMI_refData, unsigned char* d_calculateMI_tarData, float* d_calculateMI_histogram)
{
  int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
  
  int acc = threadIdx.y*BLOCK_DIM2D+threadIdx.x;
  
  __shared__ float s_diff[3];
  __shared__ float s_refDataSize[3];
  __shared__ float s_tarDataSize[3];
  __shared__ float s_refDataThickness[3];
  __shared__ float s_tarDataThickness[3];
  __shared__ float s_xdec[BLOCK_DIM2D*BLOCK_DIM2D];
  __shared__ float s_ydec[BLOCK_DIM2D*BLOCK_DIM2D];
  __shared__ float s_zdec[BLOCK_DIM2D*BLOCK_DIM2D];
  __shared__ int s_temp[BLOCK_DIM2D*BLOCK_DIM2D];
  __shared__ int s_tarDataTemp[BLOCK_DIM2D*BLOCK_DIM2D];
  
  __syncthreads();
      
  if(acc<3){
    s_refDataSize[acc]=c_calculateMI_refDataSize[acc];
    s_tarDataSize[acc]=c_calculateMI_tarDataSize[acc];
    s_refDataThickness[acc]=c_calculateMI_refDataThickness[acc];
    s_tarDataThickness[acc]=c_calculateMI_tarDataThickness[acc];
    s_diff[acc]=((c_calculateMI_refDataSize[acc]*c_calculateMI_refDataThickness[acc])-(c_calculateMI_tarDataSize[acc]*c_calculateMI_tarDataThickness[acc]))/2.0;
  }
  __syncthreads();

  int zIndex;
  float xpos, ypos, zpos;
  int xint, yint, zint;
  //float xdec, ydec, zdec;
  int outIndex;
  //int temp;
  //unsigned char tarDataTemp;

  xpos=((xIndex+0.5)*s_tarDataThickness[0]+s_diff[0])/s_refDataThickness[0]-0.5;
  ypos=((yIndex+0.5)*s_tarDataThickness[1]+s_diff[1])/s_refDataThickness[1]-0.5;

  xint=(int)xpos;
  yint=(int)ypos;

  s_xdec[acc]=xpos-xint;
  s_ydec[acc]=ypos-yint;

  if(xpos>=0 && xpos < s_refDataSize[0]-1 && ypos>=0 && ypos < s_refDataSize[1]-1){
    
    for(zIndex=0;zIndex<s_tarDataSize[2];zIndex++){
      outIndex=zIndex*s_tarDataSize[0]*s_tarDataSize[1]+yIndex*s_tarDataSize[0]+xIndex;
      
      s_tarDataTemp[acc]=d_calculateMI_tarData[outIndex];
      
      zpos=((zIndex+0.5)*s_tarDataThickness[2]+s_diff[2])/s_refDataThickness[2]-0.5;
      
      if(zpos>=0 && zpos < s_refDataSize[2]-1){
    
    zint=(int)zpos;
    
    s_zdec[acc]=zpos-zint;
    
    s_temp[acc]=zint*(int)s_refDataSize[0]*(int)s_refDataSize[1]+yint*(int)s_refDataSize[0]+xint;
    
    
    d_calculateMI_histogram[acc*256*256+(d_calculateMI_refData[s_temp[acc]])*256+s_tarDataTemp[acc]]+=(1-s_xdec[acc])*(1-s_ydec[acc])*(1-s_zdec[acc]); 
        
    d_calculateMI_histogram[acc*256*256+(d_calculateMI_refData[s_temp[acc]+(int)s_refDataSize[0]*(int)s_refDataSize[1]])*256+s_tarDataTemp[acc]]+=(1-s_xdec[acc])*(1-s_ydec[acc])*(s_zdec[acc]); 
    d_calculateMI_histogram[acc*256*256+(d_calculateMI_refData[s_temp[acc]+(int)s_refDataSize[0]])*256+s_tarDataTemp[acc]]+=(1-s_xdec[acc])*(s_ydec[acc])*(1-s_zdec[acc]); 
    d_calculateMI_histogram[acc*256*256+(d_calculateMI_refData[s_temp[acc]+(int)s_refDataSize[0]*(int)s_refDataSize[1]+(int)s_refDataSize[0]])*256+s_tarDataTemp[acc]]+=(1-s_xdec[acc])*(s_ydec[acc])*(s_zdec[acc]); 
    d_calculateMI_histogram[acc*256*256+(d_calculateMI_refData[s_temp[acc]+1])*256+s_tarDataTemp[acc]]+=(s_xdec[acc])*(1-s_ydec[acc])*(1-s_zdec[acc]); 
    d_calculateMI_histogram[acc*256*256+(d_calculateMI_refData[s_temp[acc]+(int)s_refDataSize[0]*(int)s_refDataSize[1]+1])*256+s_tarDataTemp[acc]]+=(s_xdec[acc])*(1-s_ydec[acc])*(s_zdec[acc]); 
    d_calculateMI_histogram[acc*256*256+(d_calculateMI_refData[s_temp[acc]+(int)s_refDataSize[0]+1])*256+s_tarDataTemp[acc]]+=(s_xdec[acc])*(s_ydec[acc])*(1-s_zdec[acc]); 
    d_calculateMI_histogram[acc*256*256+(d_calculateMI_refData[s_temp[acc]+(int)s_refDataSize[0]*(int)s_refDataSize[1]+(int)s_refDataSize[0]+1])*256+s_tarDataTemp[acc]]+=(s_xdec[acc])*(s_ydec[acc])*(s_zdec[acc]);
    
    /*
    d_calculateMI_histogram[acc*256*256+(d_calculateMI_refData[temp])*256+tarDataTemp]+=(1-xdec)*(1-ydec)*(1-zdec); 
    
    d_calculateMI_histogram[acc*256*256+(d_calculateMI_refData[temp+(int)s_refDataSize[0]*(int)s_refDataSize[1]])*256+tarDataTemp]+=(1-xdec)*(1-ydec)*(zdec); 
    d_calculateMI_histogram[acc*256*256+(d_calculateMI_refData[temp+(int)s_refDataSize[0]])*256+tarDataTemp]+=(1-xdec)*(ydec)*(1-zdec); 
    d_calculateMI_histogram[acc*256*256+(d_calculateMI_refData[temp+(int)s_refDataSize[0]*(int)s_refDataSize[1]+(int)s_refDataSize[0]])*256+tarDataTemp]+=(1-xdec)*(ydec)*(zdec); 
    d_calculateMI_histogram[acc*256*256+(d_calculateMI_refData[temp+1])*256+tarDataTemp]+=(xdec)*(1-ydec)*(1-zdec); 
    d_calculateMI_histogram[acc*256*256+(d_calculateMI_refData[temp+(int)s_refDataSize[0]*(int)s_refDataSize[1]+1])*256+tarDataTemp]+=(xdec)*(1-ydec)*(zdec); 
    d_calculateMI_histogram[acc*256*256+(d_calculateMI_refData[temp+(int)s_refDataSize[0]+1])*256+tarDataTemp]+=(xdec)*(ydec)*(1-zdec); 
    d_calculateMI_histogram[acc*256*256+(d_calculateMI_refData[temp+(int)s_refDataSize[0]*(int)s_refDataSize[1]+(int)s_refDataSize[0]+1])*256+tarDataTemp]+=(xdec)*(ydec)*(zdec);
    */
      }else{
    s_temp[acc]=-1;
      }
      /*
      __syncthreads();
      
      if(acc==0){
    for(int i=0;i<BLOCK_DIM2D*BLOCK_DIM2D;i++){
      if(s_temp[i]!=-1){
        d_calculateMI_histogram[i*256*256+(d_calculateMI_refData[s_temp[i]])*256+s_tarDataTemp[i]]+=(1-s_xdec[i])*(1-s_ydec[i])*(1-s_zdec[i]); 
        
        d_calculateMI_histogram[i*256*256+(d_calculateMI_refData[s_temp[i]+(int)s_refDataSize[0]*(int)s_refDataSize[1]])*256+s_tarDataTemp[i]]+=(1-s_xdec[i])*(1-s_ydec[i])*(s_zdec[i]); 
        d_calculateMI_histogram[i*256*256+(d_calculateMI_refData[s_temp[i]+(int)s_refDataSize[0]])*256+s_tarDataTemp[i]]+=(1-s_xdec[i])*(s_ydec[i])*(1-s_zdec[i]); 
        d_calculateMI_histogram[i*256*256+(d_calculateMI_refData[s_temp[i]+(int)s_refDataSize[0]*(int)s_refDataSize[1]+(int)s_refDataSize[0]])*256+s_tarDataTemp[i]]+=(1-s_xdec[i])*(s_ydec[i])*(s_zdec[i]); 
        d_calculateMI_histogram[i*256*256+(d_calculateMI_refData[s_temp[i]+1])*256+s_tarDataTemp[i]]+=(s_xdec[i])*(1-s_ydec[i])*(1-s_zdec[i]); 
        d_calculateMI_histogram[i*256*256+(d_calculateMI_refData[s_temp[i]+(int)s_refDataSize[0]*(int)s_refDataSize[1]+1])*256+s_tarDataTemp[i]]+=(s_xdec[i])*(1-s_ydec[i])*(s_zdec[i]); 
        d_calculateMI_histogram[i*256*256+(d_calculateMI_refData[s_temp[i]+(int)s_refDataSize[0]+1])*256+s_tarDataTemp[i]]+=(s_xdec[i])*(s_ydec[i])*(1-s_zdec[i]); 
        d_calculateMI_histogram[i*256*256+(d_calculateMI_refData[s_temp[i]+(int)s_refDataSize[0]*(int)s_refDataSize[1]+(int)s_refDataSize[0]+1])*256+s_tarDataTemp[i]]+=(s_xdec[i])*(s_ydec[i])*(s_zdec[i]);
        
      }
    }
      }
      
      __syncthreads();
      */
      }
  }else{
    for(zIndex=0;zIndex<s_tarDataSize[2];zIndex++){
      /*
      __syncthreads();
      __syncthreads();
      */
    }
  }
  
  __syncthreads();
  
}

__global__ void CUDAkernel_calculateMI_fillHistogram_default(unsigned char* d_calculateMI_refData, unsigned char* d_calculateMI_tarData, float* d_calculateMI_histogram)
{
  int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
  
  int acc = threadIdx.y*BLOCK_DIM2D+threadIdx.x;
  
  __shared__ float s_diff[3];
  __shared__ float s_refDataSize[3];
  __shared__ float s_tarDataSize[3];
  __shared__ float s_refDataThickness[3];
  __shared__ float s_tarDataThickness[3];
  __shared__ float s_xdec[BLOCK_DIM2D*BLOCK_DIM2D];
  __shared__ float s_ydec[BLOCK_DIM2D*BLOCK_DIM2D];
  __shared__ float s_zdec[BLOCK_DIM2D*BLOCK_DIM2D];
  __shared__ int s_temp[BLOCK_DIM2D*BLOCK_DIM2D];
  __shared__ int s_tarDataTemp[BLOCK_DIM2D*BLOCK_DIM2D];
  
  __syncthreads();
      
  if(acc<3){
    s_refDataSize[acc]=c_calculateMI_refDataSize[acc];
    s_tarDataSize[acc]=c_calculateMI_tarDataSize[acc];
    s_refDataThickness[acc]=c_calculateMI_refDataThickness[acc];
    s_tarDataThickness[acc]=c_calculateMI_tarDataThickness[acc];
    s_diff[acc]=((c_calculateMI_refDataSize[acc]*c_calculateMI_refDataThickness[acc])-(c_calculateMI_tarDataSize[acc]*c_calculateMI_tarDataThickness[acc]))/2.0;
  }
  __syncthreads();

  int zIndex;
  float xpos, ypos, zpos;
  int xint, yint, zint;
  //float xdec, ydec, zdec;
  int outIndex;
  //int temp;
  //unsigned char tarDataTemp;

  xpos=((xIndex+0.5)*s_tarDataThickness[0]+s_diff[0])/s_refDataThickness[0]-0.5;
  ypos=((yIndex+0.5)*s_tarDataThickness[1]+s_diff[1])/s_refDataThickness[1]-0.5;

  xint=(int)xpos;
  yint=(int)ypos;

  s_xdec[acc]=xpos-xint;
  s_ydec[acc]=ypos-yint;

  if(xpos>=0 && xpos < s_refDataSize[0]-1 && ypos>=0 && ypos < s_refDataSize[1]-1){
    
    for(zIndex=0;zIndex<s_tarDataSize[2];zIndex++){
      outIndex=zIndex*s_tarDataSize[0]*s_tarDataSize[1]+yIndex*s_tarDataSize[0]+xIndex;
      
      s_tarDataTemp[acc]=d_calculateMI_tarData[outIndex];
      
      zpos=((zIndex+0.5)*s_tarDataThickness[2]+s_diff[2])/s_refDataThickness[2]-0.5;
      
      if(zpos>=0 && zpos < s_refDataSize[2]-1){
    
    zint=(int)zpos;
    
    s_zdec[acc]=zpos-zint;
    
    s_temp[acc]=zint*(int)s_refDataSize[0]*(int)s_refDataSize[1]+yint*(int)s_refDataSize[0]+xint;
    
    d_calculateMI_histogram[acc*256*256+(d_calculateMI_refData[s_temp[acc]])*256+s_tarDataTemp[acc]]+=(1-s_xdec[acc])*(1-s_ydec[acc])*(1-s_zdec[acc]); 
        
    d_calculateMI_histogram[acc*256*256+(d_calculateMI_refData[s_temp[acc]+(int)s_refDataSize[0]*(int)s_refDataSize[1]])*256+s_tarDataTemp[acc]]+=(1-s_xdec[acc])*(1-s_ydec[acc])*(s_zdec[acc]); 
    d_calculateMI_histogram[acc*256*256+(d_calculateMI_refData[s_temp[acc]+(int)s_refDataSize[0]])*256+s_tarDataTemp[acc]]+=(1-s_xdec[acc])*(s_ydec[acc])*(1-s_zdec[acc]); 
    d_calculateMI_histogram[acc*256*256+(d_calculateMI_refData[s_temp[acc]+(int)s_refDataSize[0]*(int)s_refDataSize[1]+(int)s_refDataSize[0]])*256+s_tarDataTemp[acc]]+=(1-s_xdec[acc])*(s_ydec[acc])*(s_zdec[acc]); 
    d_calculateMI_histogram[acc*256*256+(d_calculateMI_refData[s_temp[acc]+1])*256+s_tarDataTemp[acc]]+=(s_xdec[acc])*(1-s_ydec[acc])*(1-s_zdec[acc]); 
    d_calculateMI_histogram[acc*256*256+(d_calculateMI_refData[s_temp[acc]+(int)s_refDataSize[0]*(int)s_refDataSize[1]+1])*256+s_tarDataTemp[acc]]+=(s_xdec[acc])*(1-s_ydec[acc])*(s_zdec[acc]); 
    d_calculateMI_histogram[acc*256*256+(d_calculateMI_refData[s_temp[acc]+(int)s_refDataSize[0]+1])*256+s_tarDataTemp[acc]]+=(s_xdec[acc])*(s_ydec[acc])*(1-s_zdec[acc]); 
    d_calculateMI_histogram[acc*256*256+(d_calculateMI_refData[s_temp[acc]+(int)s_refDataSize[0]*(int)s_refDataSize[1]+(int)s_refDataSize[0]+1])*256+s_tarDataTemp[acc]]+=(s_xdec[acc])*(s_ydec[acc])*(s_zdec[acc]);
    
    /*
    d_calculateMI_histogram[acc*256*256+(d_calculateMI_refData[temp])*256+tarDataTemp]+=(1-xdec)*(1-ydec)*(1-zdec); 
    
    d_calculateMI_histogram[acc*256*256+(d_calculateMI_refData[temp+(int)s_refDataSize[0]*(int)s_refDataSize[1]])*256+tarDataTemp]+=(1-xdec)*(1-ydec)*(zdec); 
    d_calculateMI_histogram[acc*256*256+(d_calculateMI_refData[temp+(int)s_refDataSize[0]])*256+tarDataTemp]+=(1-xdec)*(ydec)*(1-zdec); 
    d_calculateMI_histogram[acc*256*256+(d_calculateMI_refData[temp+(int)s_refDataSize[0]*(int)s_refDataSize[1]+(int)s_refDataSize[0]])*256+tarDataTemp]+=(1-xdec)*(ydec)*(zdec); 
    d_calculateMI_histogram[acc*256*256+(d_calculateMI_refData[temp+1])*256+tarDataTemp]+=(xdec)*(1-ydec)*(1-zdec); 
    d_calculateMI_histogram[acc*256*256+(d_calculateMI_refData[temp+(int)s_refDataSize[0]*(int)s_refDataSize[1]+1])*256+tarDataTemp]+=(xdec)*(1-ydec)*(zdec); 
    d_calculateMI_histogram[acc*256*256+(d_calculateMI_refData[temp+(int)s_refDataSize[0]+1])*256+tarDataTemp]+=(xdec)*(ydec)*(1-zdec); 
    d_calculateMI_histogram[acc*256*256+(d_calculateMI_refData[temp+(int)s_refDataSize[0]*(int)s_refDataSize[1]+(int)s_refDataSize[0]+1])*256+tarDataTemp]+=(xdec)*(ydec)*(zdec);
    */
      }
    }
  }
  
  __syncthreads();
  
}



__global__ void CUDAkernel_calculateMI_integrateHistogram(float* d_calculateMI_histogram)
{
  int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
  
  int acc = yIndex*256+xIndex;
  int i;
  
  for(i=1;i<BLOCK_DIM2D*BLOCK_DIM2D;i++){
    d_calculateMI_histogram[acc]+=d_calculateMI_histogram[acc+i*256*256];
  }
}

__global__ void CUDAkernel_calculateMI_calculateValue(float* d_calculateMI_histogram, float* d_calculateMI_value)
{
  int xIndex = blockDim.x * blockIdx.x + threadIdx.x;

  __shared__ float s_value[256];
  __shared__ float s_valueA[256];
  __shared__ float s_valueB[256];
  __shared__ float s_histogram[256];
  __shared__ float s_histogramA[256];
  __shared__ float s_histogramB[256];
  __shared__ int s_count[256];

  s_value[xIndex]=0;
  s_valueA[xIndex]=0;
  s_valueB[xIndex]=0;
  s_histogram[xIndex]=0;
  s_histogramA[xIndex]=0;
  s_histogramB[xIndex]=0;
  s_count[xIndex]=0;

  __syncthreads();

  int yIndex;
    
  for(yIndex=0;yIndex<256;yIndex++){
    s_histogram[xIndex]=d_calculateMI_histogram[yIndex*256+xIndex];
    s_histogramA[xIndex]+=s_histogram[xIndex];
    //s_histogramB[yIndex]+=s_histogram[xIndex];
    s_histogramB[xIndex]+=d_calculateMI_histogram[xIndex*256+yIndex];
    s_count[xIndex]+=s_histogram[xIndex];
  }
    
  __syncthreads();

  if(xIndex<128){
    s_count[xIndex*2]+=s_count[xIndex*2+1];
  }
  
  __syncthreads();

  if(xIndex<64){
    s_count[xIndex*4]+=s_count[xIndex*4+2];
  }
  
  __syncthreads();

  if(xIndex<32){
    s_count[xIndex*8]+=s_count[xIndex*8+4];
  }

  __syncthreads();

  if(xIndex<16){
    s_count[xIndex*16]+=s_count[xIndex*16+8];
  }

  __syncthreads();

  if(xIndex<8){
    s_count[xIndex*32]+=s_count[xIndex*32+16];
  }

  __syncthreads();
  
  if(xIndex<4){
    s_count[xIndex*64]+=s_count[xIndex*64+32];
  }

  __syncthreads();
  
  if(xIndex<2){
    s_count[xIndex*128]+=s_count[xIndex*128+64];
  }

  __syncthreads();

  if(xIndex<1){
    s_count[0]+=s_count[128];
  }

  __syncthreads();
  
  float tempval;

  if(s_histogramA[xIndex]!=0){
    tempval=s_histogramA[xIndex]/(float)s_count[0];
    s_valueA[xIndex]=-tempval*log(tempval)/log(10.0);
  }
  if(s_histogramB[xIndex]!=0){
    tempval=s_histogramB[xIndex]/(float)s_count[0];
    s_valueB[xIndex]=-tempval*log(tempval)/log(10.0);
  }
  
  for(yIndex=0;yIndex<256;yIndex++){
    s_histogram[xIndex]=d_calculateMI_histogram[yIndex*256+xIndex];
    if(s_histogram[xIndex]!=0){
      tempval=s_histogram[xIndex]/(float)s_count[0];
      s_value[xIndex]+=tempval*log(tempval)/log(10.0);
    }
  }

  __syncthreads();

  if(xIndex<128){
    s_value[xIndex*2]+=s_value[xIndex*2+1];
    s_valueA[xIndex*2]+=s_valueA[xIndex*2+1];
    s_valueB[xIndex*2]+=s_valueB[xIndex*2+1];
  }
  
  __syncthreads();

  if(xIndex<64){
    s_value[xIndex*4]+=s_value[xIndex*4+2];
    s_valueA[xIndex*4]+=s_valueA[xIndex*4+2];
    s_valueB[xIndex*4]+=s_valueB[xIndex*4+2];
  }
  
  __syncthreads();

  if(xIndex<32){
    s_value[xIndex*8]+=s_value[xIndex*8+4];
    s_valueA[xIndex*8]+=s_valueA[xIndex*8+4];
    s_valueB[xIndex*8]+=s_valueB[xIndex*8+4];
  }

  __syncthreads();

  if(xIndex<16){
    s_value[xIndex*16]+=s_value[xIndex*16+8];
    s_valueA[xIndex*16]+=s_valueA[xIndex*16+8];
    s_valueB[xIndex*16]+=s_valueB[xIndex*16+8];
  }

  __syncthreads();

  if(xIndex<8){
    s_value[xIndex*32]+=s_value[xIndex*32+16];
    s_valueA[xIndex*32]+=s_valueA[xIndex*32+16];
    s_valueB[xIndex*32]+=s_valueB[xIndex*32+16];
  }

  __syncthreads();
  
  if(xIndex<4){
    s_value[xIndex*64]+=s_value[xIndex*64+32];
    s_valueA[xIndex*64]+=s_valueA[xIndex*64+32];
    s_valueB[xIndex*64]+=s_valueB[xIndex*64+32];
  }

  __syncthreads();
  
  if(xIndex<2){
    s_value[xIndex*128]+=s_value[xIndex*128+64];
    s_valueA[xIndex*128]+=s_valueA[xIndex*128+64];
    s_valueB[xIndex*128]+=s_valueB[xIndex*128+64];
  }

  __syncthreads();
  
  if(xIndex<1){
    s_value[0]+=s_value[128];
    s_valueA[0]+=s_valueA[128];
    s_valueB[0]+=s_valueB[128];
  }
  
  __syncthreads();
  
  if(xIndex<1){
    //d_calculateMI_value[0]=s_count[0];
    d_calculateMI_value[0]=s_value[0];
    d_calculateMI_value[1]=s_valueA[0];
    d_calculateMI_value[2]=s_valueB[0];
  }

  
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////extern "C"
double CUDAcalculateMI_doCalculation(unsigned char* refData, unsigned char* tarData, int refSizeX, int refSizeY, int refSizeZ, int tarSizeX, int tarSizeY, int tarSizeZ, float refThicknessX, float refThicknessY, float refThicknessZ, float tarThicknessX, float tarThicknessY, float tarThicknessZ)
{
  // size of the matrix

  // setup execution parameters
  /*
  dim3 grid(tarSizeX / BLOCK_DIM3D, tarSizeY / BLOCK_DIM3D * tarSizeZ /BLOCK_DIM3D, 1);
  dim3 threads(BLOCK_DIM3D, BLOCK_DIM3D, BLOCK_DIM3D);
  */
  dim3 grid(tarSizeX / BLOCK_DIM2D, tarSizeY / BLOCK_DIM2D, 1);
  dim3 threads(BLOCK_DIM2D, BLOCK_DIM2D, 1);

  dim3 grid2(256 / BLOCK_DIM2D, 256 / BLOCK_DIM2D, 1);
  dim3 threads2(BLOCK_DIM2D, BLOCK_DIM2D, 1);

  dim3 grid3(256 / 16, 256 / 16, 1);
  dim3 threads3(16, 16, 1);

  dim3 grid4(256 / BLOCK_DIM1D, 1, 1);
  dim3 threads4(BLOCK_DIM1D, 1, 1);

#ifdef USE_TIMER

  unsigned int timer;
  cutCreateTimer(&timer);

  cutStartTimer(timer);

#endif
  CUT_DEVICE_INIT();

  float refDataThickness[3]={refThicknessX, refThicknessY, refThicknessZ};
  float tarDataThickness[3]={tarThicknessX, tarThicknessY, tarThicknessZ};
  float tarDataSize[3]={tarSizeX, tarSizeY, tarSizeZ};
  float refDataSize[3]={refSizeX, refSizeY, refSizeZ};
  
  CUDA_SAFE_CALL( cudaMemcpyToSymbol(c_calculateMI_refDataSize, refDataSize, sizeof(float)*3, 0));
  CUDA_SAFE_CALL( cudaMemcpyToSymbol(c_calculateMI_refDataThickness, refDataThickness, sizeof(float)*3, 0));
  CUDA_SAFE_CALL( cudaMemcpyToSymbol(c_calculateMI_tarDataThickness, tarDataThickness, sizeof(float)*3, 0));
  CUDA_SAFE_CALL( cudaMemcpyToSymbol(c_calculateMI_tarDataSize, tarDataSize, sizeof(float)*3, 0));
  CUDA_SAFE_CALL( cudaMalloc( (void**) &d_calculateMI_value, sizeof(float)*3));

  CUDA_SAFE_CALL( cudaMalloc( (void**) &d_calculateMI_histogram, sizeof(float)*256*256*BLOCK_DIM2D*BLOCK_DIM2D));

  CUDA_SAFE_CALL( cudaMallocHost( (void**) &h_calculateMI_histogram, sizeof(float)*256*256*BLOCK_DIM2D*BLOCK_DIM2D));
  CUDA_SAFE_CALL( cudaMallocHost( (void**) &h_calculateMI_value, sizeof(float)*3));

#ifdef USE_TIMER
  cutStopTimer(timer);
  float naiveTime = cutGetTimerValue(timer);
  printf("Memory copy CPU to GPU average time:     %0.3f ms\n", naiveTime);fflush(stdout);
  cutResetTimer(timer);
  cutStartTimer(timer);
#endif

  // execute the kernel
  CUDAkernel_calculateMI_initHistogram<<< grid2, threads2 >>>(d_calculateMI_histogram);

  CUDAkernel_calculateMI_fillHistogram<<< grid, threads >>>(refData, tarData, d_calculateMI_histogram);
  
  CUDAkernel_calculateMI_integrateHistogram<<< grid3, threads3 >>>(d_calculateMI_histogram);

  CUDAkernel_calculateMI_calculateValue<<< grid4, threads4 >>>(d_calculateMI_histogram, d_calculateMI_value);

  CUDA_SAFE_CALL( cudaMemcpy(h_calculateMI_value, d_calculateMI_value, sizeof(float)*3,cudaMemcpyDeviceToHost));

  //printf("MI value= %lf\n", h_calculateMI_value[0]+h_calculateMI_value[1]+h_calculateMI_value[2]);

  printf("%lf %lf %lf\n", h_calculateMI_value[0], h_calculateMI_value[1], h_calculateMI_value[2]);

  float mival=h_calculateMI_value[0]+h_calculateMI_value[1]+h_calculateMI_value[2];

  CUT_CHECK_ERROR("Kernel execution failed");

#ifdef USE_TIMER
  cutStopTimer(timer);
  naiveTime = cutGetTimerValue(timer);
  printf("calculate MI average time:     %0.3f ms\n", naiveTime);
#endif

  CUDA_SAFE_CALL(cudaFree(d_calculateMI_histogram));
  CUDA_SAFE_CALL(cudaFree(d_calculateMI_value));

  CUDA_SAFE_CALL(cudaFreeHost(h_calculateMI_histogram));

  return mival;
}

