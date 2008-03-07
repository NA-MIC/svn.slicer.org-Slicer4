// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil.h>

// includes, kernels
#include "CUDA_calculateSSDGradient.h"

//#define USE_TIMER
#define BLOCK_DIM2D 8// this must be set to 4 or more
#define ACC(X,Y,Z) ( ( (Z)*(sizeX)*(sizeY) ) + ( (Y)*(sizeX) ) + (X) )
#define SQR(X) ((X) * (X) )

__constant__ float c_calculateSSDGradient_refDataSize[3];
__constant__ float c_calculateSSDGradient_tarDataSize[3];
__constant__ float c_calculateSSDGradient_splineSize[3];

__device__ float* d_newX;
__device__ float* d_newY;
__device__ float* d_newZ;
__device__ float* d_diff;

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

__device__ float getDifferentialValue(float x){
  float result=0.0;
  float xabs=fabs(x);
  if(xabs<=1.0){
    result=(2.25*xabs*xabs-3*xabs);
  }else if(xabs<2.0){
    result=(-0.75*(2.0-xabs)*(2.0-xabs));
  }else{
    return 0;
  }
  
  if(x<0){
    result=0.0-result;
  }
  return result;
}

__global__ void CUDAkernel_calculateSSDGradient_preparation(unsigned char* referenceData, unsigned char* targetData, float* transSpline, float* newX, float* newY, float* newZ, float* diff, int tempPow)
{
  int xIndex = (blockDim.x*blockIdx.x + threadIdx.x) % (int)c_calculateSSDGradient_refDataSize[0];
  int yIndex = (blockDim.x*blockIdx.x + threadIdx.x) / (int)c_calculateSSDGradient_refDataSize[0];
  int zIndex = blockDim.y*blockIdx.y+ threadIdx.y;
  
  
  __shared__ float s_refDataSize[3];
  __shared__ float s_splineSize[3];
  __shared__ float s_newX[BLOCK_DIM2D*BLOCK_DIM2D];
  __shared__ float s_newY[BLOCK_DIM2D*BLOCK_DIM2D];
  __shared__ float s_newZ[BLOCK_DIM2D*BLOCK_DIM2D];
   
  __shared__ unsigned char s_outputData[BLOCK_DIM2D*BLOCK_DIM2D];
  
  int acc=threadIdx.y*BLOCK_DIM2D+threadIdx.x;

  if(acc<3){
    s_refDataSize[acc]=c_calculateSSDGradient_refDataSize[acc];
  }else if(acc<6){
    s_splineSize[acc-3]=c_calculateSSDGradient_splineSize[acc-3];
  }

  __syncthreads();

  int outIndex=zIndex*(int)s_refDataSize[0]*(int)s_refDataSize[1]+yIndex*(int)s_refDataSize[0]+xIndex;
  
  float relPosX, relPosY, relPosZ;
  float functemp;
  float tempValue;
  int a,b,c;
  
  s_newX[acc]=xIndex; s_newY[acc]=yIndex; s_newZ[acc]=zIndex; 
  
  __syncthreads();
  
  if(xIndex<s_refDataSize[0] && yIndex < s_refDataSize[1] && zIndex < s_refDataSize[2]){

    relPosX=(float)(xIndex+1)*tempPow;
    relPosY=(float)(yIndex+1)*tempPow;
    relPosZ=(float)(zIndex+1)*tempPow;
    for(a=(int)relPosX-1;a<(int)relPosX+3;a++){
      if(a<0||a>s_splineSize[0]-1)continue;
      for(b=(int)relPosY-1;b<(int)relPosY+3;b++){
        if(b<0||b>s_splineSize[1]-1)continue;
  for(c=(int)relPosZ-1;c<(int)relPosZ+3;c++){
    if(c<0||c>s_splineSize[2]-1)continue;
    
    //if(a>=0 && a<=s_splineSize[0]-1 && b>=0 && b<=s_splineSize[1]-1 && c>=0 && c<=s_splineSize[2]-1 ){
      functemp=getSplineValue(relPosX-(float)a)*getSplineValue(relPosY-(float)b)*getSplineValue(relPosZ-(float)c);
      s_newX[acc]+=transSpline[c*(int)s_splineSize[1]*(int)s_splineSize[0]+b*(int)s_splineSize[0]+a]*functemp;
      s_newY[acc]+=transSpline[(int)s_splineSize[0]*(int)s_splineSize[1]*(int)s_splineSize[2]+c*(int)s_splineSize[1]*(int)s_splineSize[0]+b*(int)s_splineSize[0]+a]*functemp;
      s_newZ[acc]+=transSpline[2*(int)s_splineSize[0]*(int)s_splineSize[1]*(int)s_splineSize[2]+c*(int)s_splineSize[1]*(int)s_splineSize[0]+b*(int)s_splineSize[0]+a]*functemp;
      //}else{
      //functemp=0;
      //}
        
  }
      }
    }
    

    if(s_newX[acc]>=0 && s_newX[acc]<=s_refDataSize[0]-1 && s_newY[acc]>=0 && s_newY[acc] <= s_refDataSize[1]-1 && s_newZ[acc]>=0 && s_newZ[acc] <= s_refDataSize[2]-1){
      tempValue=targetData[(__float2int_rn(s_newZ[acc]))*(int)s_refDataSize[0]*(int)s_refDataSize[1]+(__float2int_rn(s_newY[acc]))*(int)s_refDataSize[0]+(__float2int_rn(s_newX[acc]))];
    }else{
      tempValue=0;
    }
    
    s_outputData[acc]=tempValue;

  }else{
    s_outputData[acc]=0;
  }
  
  if(xIndex<s_refDataSize[0] && yIndex < s_refDataSize[1] && zIndex < s_refDataSize[2]){
    newX[outIndex]=s_newX[acc];
    newY[outIndex]=s_newY[acc];
    newZ[outIndex]=s_newZ[acc];
    diff[outIndex]=s_outputData[acc]-referenceData[outIndex];
  }

  return;
}

__global__ void CUDAkernel_calculateSSDGradient_doCalculation(unsigned char* referenceData, unsigned char* targetData, float* transSpline, float* SSDGradient, float* newX, float* newY, float* newZ, float* diff, int tempPow)
{
  int xIndex = (blockDim.x*blockIdx.x + threadIdx.x) % (int)c_calculateSSDGradient_splineSize[0];
  int yIndex = (blockDim.x*blockIdx.x + threadIdx.x) / (int)c_calculateSSDGradient_splineSize[0];
  int zIndex = blockDim.y*blockIdx.y+ threadIdx.y;
  
  __shared__ float s_refDataSize[3];
  __shared__ float s_tarDataSize[3];
  __shared__ float s_splineSize[3];
  __shared__ float s_SSDGradient[BLOCK_DIM2D*BLOCK_DIM2D*3];
     
  int acc=threadIdx.y*BLOCK_DIM2D+threadIdx.x;

  if(acc<3){
    s_refDataSize[acc]=c_calculateSSDGradient_refDataSize[acc];
    s_tarDataSize[acc]=c_calculateSSDGradient_tarDataSize[acc];
  }else if(acc<6){
    s_splineSize[acc-3]=c_calculateSSDGradient_splineSize[acc-3];
  }

  __syncthreads();

  int outIndex=zIndex*(int)s_splineSize[0]*(int)s_splineSize[1]+yIndex*(int)s_splineSize[0]+xIndex;
  
  s_SSDGradient[acc*3]=0;
  s_SSDGradient[acc*3+1]=0;
  s_SSDGradient[acc*3+2]=0;

  __syncthreads();
 
  int i, j, k;
  int d, e, f;
  float knotDistance=1.0f/(float)tempPow;
  int targetIndex;
  float val1, val21, val22, val23;
  float relPosX, relPosY, relPosZ;
  
  
  if(xIndex < s_splineSize[0] && yIndex < s_splineSize[1] && zIndex < s_splineSize[2]){
    
    for(i=((xIndex-2)*knotDistance-1);i<((xIndex+2)*knotDistance);i++){
      if(i<0 || i>s_tarDataSize[0]-1) continue;
      for(j=((yIndex-2)*knotDistance-1);j<((yIndex+2)*knotDistance);j++){
  if(j<0 || j>s_tarDataSize[1]-1) continue;
  for(k=((zIndex-2)*knotDistance-1);k<((zIndex+2)*knotDistance);k++){
    if(k<0 || k>s_tarDataSize[2]-1) continue;
    
    //calculation of SSD gradient against x axis deformation parameter
    
    targetIndex=k*s_tarDataSize[0]*s_tarDataSize[1]+j*s_tarDataSize[0]+i;
    
    
    val1=diff[targetIndex];
    
    val21=0;    
    val22=0;    
    val23=0;    
    relPosX=newX[targetIndex]+1.0;
    relPosY=newY[targetIndex]+1.0;
    relPosZ=newZ[targetIndex]+1.0;
    
    for(d=(int)relPosX-1;d<(int)relPosX+3;d++){
      if(d<1||d>s_tarDataSize[0])continue;
      for(e=(int)relPosY-1;e<(int)relPosY+3;e++){
        if(e<1||e>s_tarDataSize[1])continue;
        for(f=(int)relPosZ-1;f<(int)relPosZ+3;f++){
    if(f<1||f>s_tarDataSize[2])continue;   
    val21+=targetData[(f-1)*(int)s_tarDataSize[0]*(int)s_tarDataSize[1]+(e-1)*(int)s_tarDataSize[0]+d-1]*getDifferentialValue(relPosX-(float)d)*getSplineValue(relPosY-(float)e)*getSplineValue(relPosZ-(float)f);
    val22+=targetData[(f-1)*(int)s_tarDataSize[0]*(int)s_tarDataSize[1]+(e-1)*(int)s_tarDataSize[0]+d-1]*getSplineValue(relPosX-(float)d)*getDifferentialValue(relPosY-(float)e)*getSplineValue(relPosZ-(float)f);
    val23+=targetData[(f-1)*(int)s_tarDataSize[0]*(int)s_tarDataSize[1]+(e-1)*(int)s_tarDataSize[0]+d-1]*getSplineValue(relPosX-(float)d)*getSplineValue(relPosY-(float)e)*getDifferentialValue(relPosZ-(float)f);
        }   
      }
    }
    
    s_SSDGradient[acc]+=val1*val21*getSplineValue((float)i*tempPow+tempPow-(float)xIndex)*getSplineValue((float)j*tempPow+tempPow-(float)yIndex)*getSplineValue((float)k*tempPow+tempPow-(float)zIndex)/knotDistance;
    s_SSDGradient[acc+BLOCK_DIM2D*BLOCK_DIM2D]+=val1*val22*getSplineValue((float)i*tempPow+tempPow-(float)xIndex)*getSplineValue((float)j*tempPow+tempPow-(float)yIndex)*getSplineValue((float)k*tempPow+tempPow-(float)zIndex)/knotDistance;
    s_SSDGradient[acc+2*BLOCK_DIM2D*BLOCK_DIM2D]+=val1*val23*getSplineValue((float)i*tempPow+tempPow-(float)xIndex)*getSplineValue((float)j*tempPow+tempPow-(float)yIndex)*getSplineValue((float)k*tempPow+tempPow-(float)zIndex)/knotDistance;
    
  }
      }
    }
  }
  
  __syncthreads();

  if(xIndex < s_splineSize[0] && yIndex < s_splineSize[1] && zIndex < s_splineSize[2]){
    SSDGradient[outIndex]=s_SSDGradient[acc]/(s_tarDataSize[0]*s_tarDataSize[1]*s_tarDataSize[2]);
    SSDGradient[outIndex+(int)s_splineSize[0]*(int)s_splineSize[1]*(int)s_splineSize[2]]=s_SSDGradient[acc+BLOCK_DIM2D*BLOCK_DIM2D]/(s_tarDataSize[0]*s_tarDataSize[1]*s_tarDataSize[2]);
    SSDGradient[outIndex+2*(int)s_splineSize[0]*(int)s_splineSize[1]*(int)s_splineSize[2]]=s_SSDGradient[acc+2*BLOCK_DIM2D*BLOCK_DIM2D]/(s_tarDataSize[0]*s_tarDataSize[1]*s_tarDataSize[2]);
  }
  
  return;
}

__global__ void CUDAkernel_calculateSSDGradient_doCalculationX(unsigned char* referenceData, unsigned char* targetData, float* transSpline, float* SSDGradient, float* newX, float* newY, float* newZ, float* diff, int tempPow)
{
  int xIndex = (blockDim.x*blockIdx.x + threadIdx.x) % (int)c_calculateSSDGradient_splineSize[0];
  int yIndex = (blockDim.x*blockIdx.x + threadIdx.x) / (int)c_calculateSSDGradient_splineSize[0];
  int zIndex = blockDim.y*blockIdx.y+ threadIdx.y;
  
  __shared__ float s_refDataSize[3];
  __shared__ float s_tarDataSize[3];
  __shared__ float s_splineSize[3];
  __shared__ float s_SSDGradient[BLOCK_DIM2D*BLOCK_DIM2D*3];
     
  int acc=threadIdx.y*BLOCK_DIM2D+threadIdx.x;

  if(acc<3){
    s_refDataSize[acc]=c_calculateSSDGradient_refDataSize[acc];
    s_tarDataSize[acc]=c_calculateSSDGradient_tarDataSize[acc];
  }else if(acc<6){
    s_splineSize[acc-3]=c_calculateSSDGradient_splineSize[acc-3];
  }

  __syncthreads();

  int outIndex=zIndex*(int)s_splineSize[0]*(int)s_splineSize[1]+yIndex*(int)s_splineSize[0]+xIndex;
  
  s_SSDGradient[acc]=0;
  
  __syncthreads();
 
  int i, j, k;
  int d, e, f;
  float knotDistance=1.0f/(float)tempPow;
  int targetIndex;
  float val1, val2;
  float relPosX, relPosY, relPosZ;
  
  
  if(xIndex < s_splineSize[0] && yIndex < s_splineSize[1] && zIndex < s_splineSize[2]){
    
    for(i=((xIndex-2)*knotDistance-1);i<((xIndex+2)*knotDistance);i++){
      if(i<0 || i>s_tarDataSize[0]-1) continue;
      for(j=((yIndex-2)*knotDistance-1);j<((yIndex+2)*knotDistance);j++){
  if(j<0 || j>s_tarDataSize[1]-1) continue;
  for(k=((zIndex-2)*knotDistance-1);k<((zIndex+2)*knotDistance);k++){
    if(k<0 || k>s_tarDataSize[2]-1) continue;
    
    //calculation of SSD gradient against x axis deformation parameter
    
    targetIndex=k*s_tarDataSize[0]*s_tarDataSize[1]+j*s_tarDataSize[0]+i;
    
    
    val1=diff[targetIndex];
    
    val2=0;    
    relPosX=newX[targetIndex];
    relPosY=newY[targetIndex];
    relPosZ=newZ[targetIndex];
    
    for(d=(int)relPosX-1;d<(int)relPosX+3;d++){
      if(d<0||d>=s_tarDataSize[0]+2)continue;
      for(e=(int)relPosY-1;e<(int)relPosY+3;e++){
        if(e<0||e>=s_tarDataSize[1]+2)continue;
        for(f=(int)relPosZ-1;f<(int)relPosZ+3;f++){
    if(f<0||f>=s_tarDataSize[2]+2)continue;   
    val2+=targetData[f*(int)s_tarDataSize[0]*(int)s_tarDataSize[1]+e*(int)s_tarDataSize[0]+d]*getDifferentialValue(relPosX-(float)d)*getSplineValue(relPosY-(float)e)*getSplineValue(relPosZ-(float)f);
        }   
      }
    }
    
    s_SSDGradient[acc]+=val1*val2*getSplineValue((float)i*tempPow+tempPow-(float)xIndex)*getSplineValue((float)j*tempPow+tempPow-(float)yIndex)*getSplineValue((float)k*tempPow+tempPow-(float)zIndex)/knotDistance;
    
    
  }
      }
    }
  }
  

  __syncthreads();

  if(xIndex < s_splineSize[0] && yIndex < s_splineSize[1] && zIndex < s_splineSize[2]){
    SSDGradient[outIndex]=s_SSDGradient[acc]/(s_tarDataSize[0]*s_tarDataSize[1]*s_tarDataSize[2]);
  }
  
  return;
}

__global__ void CUDAkernel_calculateSSDGradient_doCalculationY(unsigned char* referenceData, unsigned char* targetData, float* transSpline, float* SSDGradient, float* newX, float* newY, float* newZ, float* diff, int tempPow)
{
  int xIndex = (blockDim.x*blockIdx.x + threadIdx.x) % (int)c_calculateSSDGradient_splineSize[0];
  int yIndex = (blockDim.x*blockIdx.x + threadIdx.x) / (int)c_calculateSSDGradient_splineSize[0];
  int zIndex = blockDim.y*blockIdx.y+ threadIdx.y;
  
  __shared__ float s_refDataSize[3];
  __shared__ float s_tarDataSize[3];
  __shared__ float s_splineSize[3];
  __shared__ float s_SSDGradient[BLOCK_DIM2D*BLOCK_DIM2D*3];
     
  int acc=threadIdx.y*BLOCK_DIM2D+threadIdx.x;

  if(acc<3){
    s_refDataSize[acc]=c_calculateSSDGradient_refDataSize[acc];
    s_tarDataSize[acc]=c_calculateSSDGradient_tarDataSize[acc];
  }else if(acc<6){
    s_splineSize[acc-3]=c_calculateSSDGradient_splineSize[acc-3];
  }

  __syncthreads();

  int outIndex=zIndex*(int)s_splineSize[0]*(int)s_splineSize[1]+yIndex*(int)s_splineSize[0]+xIndex;
  
  s_SSDGradient[acc+BLOCK_DIM2D*BLOCK_DIM2D]=0;
  
  __syncthreads();
 
  int i, j, k;
  int d, e, f;
  float knotDistance=1.0f/(float)tempPow;
  int targetIndex;
  float val1, val2;
  float relPosX, relPosY, relPosZ;
  
  
  if(xIndex < s_splineSize[0] && yIndex < s_splineSize[1] && zIndex < s_splineSize[2]){
    
    for(i=((xIndex-2)*knotDistance-1);i<((xIndex+2)*knotDistance);i++){
      if(i<0 || i>s_tarDataSize[0]-1) continue;
      for(j=((yIndex-2)*knotDistance-1);j<((yIndex+2)*knotDistance);j++){
  if(j<0 || j>s_tarDataSize[1]-1) continue;
  for(k=((zIndex-2)*knotDistance-1);k<((zIndex+2)*knotDistance);k++){
    if(k<0 || k>s_tarDataSize[2]-1) continue;
    
    //calculation of SSD gradient against x axis deformation parameter
    
    targetIndex=k*s_tarDataSize[0]*s_tarDataSize[1]+j*s_tarDataSize[0]+i;
    
    //calculation of SSD gradient against y axis deformation parameter
    val1=diff[targetIndex];
    
    val2=0;
    relPosX=newX[targetIndex];
    relPosY=newY[targetIndex];
    relPosZ=newZ[targetIndex];
    
    for(d=(int)relPosX-1;d<(int)relPosX+3;d++){
      if(d<0||d>=s_tarDataSize[0]+2)continue;
      for(e=(int)relPosY-1;e<(int)relPosY+3;e++){
        if(e<0||e>=s_tarDataSize[1]+2)continue;
        for(f=(int)relPosZ-1;f<(int)relPosZ+3;f++){
    if(f<0||f>=s_tarDataSize[2]+2)continue;   
    val2+=targetData[f*(int)s_tarDataSize[0]*(int)s_tarDataSize[1]+e*(int)s_tarDataSize[0]+d]
      *getSplineValue(relPosX-(float)d)*getDifferentialValue(relPosY-(float)e)*getSplineValue(relPosZ-(float)f);
        }   
      }
    }
    
    s_SSDGradient[acc+BLOCK_DIM2D*BLOCK_DIM2D]+=val1*val2*getSplineValue((float)i*tempPow+tempPow-(float)xIndex)*getSplineValue((float)j*tempPow+tempPow-(float)yIndex)*getSplineValue((float)k*tempPow+tempPow-(float)zIndex)/knotDistance;
      
  }
      }
    }
  }
  

  __syncthreads();

  if(xIndex < s_splineSize[0] && yIndex < s_splineSize[1] && zIndex < s_splineSize[2]){
    SSDGradient[outIndex+(int)s_splineSize[0]*(int)s_splineSize[1]*(int)s_splineSize[2]]=s_SSDGradient[acc+BLOCK_DIM2D*BLOCK_DIM2D]/(s_tarDataSize[0]*s_tarDataSize[1]*s_tarDataSize[2]);
  }
  
  return;
}

__global__ void CUDAkernel_calculateSSDGradient_doCalculationZ(unsigned char* referenceData, unsigned char* targetData, float* transSpline, float* SSDGradient, float* newX, float* newY, float* newZ, float* diff, int tempPow)
{
  int xIndex = (blockDim.x*blockIdx.x + threadIdx.x) % (int)c_calculateSSDGradient_splineSize[0];
  int yIndex = (blockDim.x*blockIdx.x + threadIdx.x) / (int)c_calculateSSDGradient_splineSize[0];
  int zIndex = blockDim.y*blockIdx.y+ threadIdx.y;
  
  __shared__ float s_refDataSize[3];
  __shared__ float s_tarDataSize[3];
  __shared__ float s_splineSize[3];
  __shared__ float s_SSDGradient[BLOCK_DIM2D*BLOCK_DIM2D*3];
     
  int acc=threadIdx.y*BLOCK_DIM2D+threadIdx.x;

  if(acc<3){
    s_refDataSize[acc]=c_calculateSSDGradient_refDataSize[acc];
    s_tarDataSize[acc]=c_calculateSSDGradient_tarDataSize[acc];
  }else if(acc<6){
    s_splineSize[acc-3]=c_calculateSSDGradient_splineSize[acc-3];
  }

  __syncthreads();

  int outIndex=zIndex*(int)s_splineSize[0]*(int)s_splineSize[1]+yIndex*(int)s_splineSize[0]+xIndex;
  
  s_SSDGradient[acc+2*BLOCK_DIM2D*BLOCK_DIM2D]=0;

  __syncthreads();
 
  int i, j, k;
  int d, e, f;
  float knotDistance=1.0f/(float)tempPow;
  int targetIndex;
  float val1, val2;
  float relPosX, relPosY, relPosZ;
  
  
  if(xIndex < s_splineSize[0] && yIndex < s_splineSize[1] && zIndex < s_splineSize[2]){
    
    for(i=((xIndex-2)*knotDistance-1);i<((xIndex+2)*knotDistance);i++){
      if(i<0 || i>s_tarDataSize[0]-1) continue;
      for(j=((yIndex-2)*knotDistance-1);j<((yIndex+2)*knotDistance);j++){
  if(j<0 || j>s_tarDataSize[1]-1) continue;
  for(k=((zIndex-2)*knotDistance-1);k<((zIndex+2)*knotDistance);k++){
    if(k<0 || k>s_tarDataSize[2]-1) continue;
    
    //calculation of SSD gradient against x axis deformation parameter
    
    targetIndex=k*s_tarDataSize[0]*s_tarDataSize[1]+j*s_tarDataSize[0]+i;
    
    //calculation of SSD gradient against z axis deformation parameter/
    
    val1=diff[targetIndex];
    
    val2=0;
    relPosX=newX[targetIndex];
    relPosY=newY[targetIndex];
    relPosZ=newZ[targetIndex];
    
    for(d=(int)relPosX-1;d<(int)relPosX+3;d++){
      if(d<0||d>=s_tarDataSize[0]+2)continue;
      for(e=(int)relPosY-1;e<(int)relPosY+3;e++){
        if(e<0||e>=s_tarDataSize[1]+2)continue;
        for(f=(int)relPosZ-1;f<(int)relPosZ+3;f++){
    if(f<0||f>=s_tarDataSize[2]+2)continue;   
    val2+=targetData[f*(int)s_tarDataSize[0]*(int)s_tarDataSize[1]+e*(int)s_tarDataSize[0]+d]
      *getSplineValue(relPosX-(float)d)*getSplineValue(relPosY-(float)e)*getDifferentialValue(relPosZ-(float)f);
        }   
      }
    }
    
    s_SSDGradient[acc+2*BLOCK_DIM2D*BLOCK_DIM2D]+=val1*val2*getSplineValue((float)i*tempPow+tempPow-(float)xIndex)*getSplineValue((float)j*tempPow+tempPow-(float)yIndex)*getSplineValue((float)k*tempPow+tempPow-(float)zIndex)/knotDistance;
    
  }
      }
    }
  }
  

  __syncthreads();

  if(xIndex < s_splineSize[0] && yIndex < s_splineSize[1] && zIndex < s_splineSize[2]){
    SSDGradient[outIndex+2*(int)s_splineSize[0]*(int)s_splineSize[1]*(int)s_splineSize[2]]=s_SSDGradient[acc+2*BLOCK_DIM2D*BLOCK_DIM2D]/(s_tarDataSize[0]*s_tarDataSize[1]*s_tarDataSize[2]);
  }
  
  return;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
extern "C"
float CUDAcalculateSSDGradient_doCalculation(unsigned char* reference, unsigned char* target, float* transSpline, float* SSDGradient, int refSizeX, int refSizeY, int refSizeZ, int splineSizeLevel)
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
  
  int splineSizeX=(int)(tempPow*(refSizeX+1)+1);
  int splineSizeY=(int)(tempPow*(refSizeY+1)+1);
  int splineSizeZ=(int)(tempPow*(refSizeZ+1)+1);

  // size of the matrix

  // setup execution parameters

  int blockX, blockY;

  blockX=((refSizeX*refSizeY-1)/BLOCK_DIM2D)+1;
  blockY=((refSizeZ-1)/BLOCK_DIM2D)+1;

  int blockSplineX, blockSplineY;

  blockSplineX=((splineSizeX*splineSizeY-1)/BLOCK_DIM2D)+1;
  blockSplineY=((splineSizeZ-1)/BLOCK_DIM2D)+1;

  dim3 grid(blockX, blockY, 1);
  dim3 threads(BLOCK_DIM2D, BLOCK_DIM2D, 1);

  dim3 gridSpline(blockSplineX, blockSplineY, 1);
  dim3 threadsSpline(BLOCK_DIM2D, BLOCK_DIM2D, 1);

  CUT_DEVICE_INIT();

  float refDataSize[3]={refSizeX, refSizeY, refSizeZ};
  CUDA_SAFE_CALL( cudaMemcpyToSymbol(c_calculateSSDGradient_refDataSize, refDataSize, sizeof(float)*3, 0));

  float tarDataSize[3]={refSizeX, refSizeY, refSizeZ};
  CUDA_SAFE_CALL( cudaMemcpyToSymbol(c_calculateSSDGradient_tarDataSize, tarDataSize, sizeof(float)*3, 0));

  float splineSize[3]={splineSizeX, splineSizeY, splineSizeZ};
  CUDA_SAFE_CALL( cudaMemcpyToSymbol(c_calculateSSDGradient_splineSize, splineSize, sizeof(float)*3, 0));

  
  CUDA_SAFE_CALL( cudaMalloc( (void**) &d_newX, sizeof(float)*refSizeX*refSizeY*refSizeZ));
  CUDA_SAFE_CALL( cudaMalloc( (void**) &d_newY, sizeof(float)*refSizeX*refSizeY*refSizeZ));
  CUDA_SAFE_CALL( cudaMalloc( (void**) &d_newZ, sizeof(float)*refSizeX*refSizeY*refSizeZ));
  CUDA_SAFE_CALL( cudaMalloc( (void**) &d_diff, sizeof(float)*refSizeX*refSizeY*refSizeZ));
  
  float * buffer = (float*)malloc(splineSizeX*splineSizeY*splineSizeZ*3*sizeof(float));


  CUDAkernel_calculateSSDGradient_preparation<<< grid, threads >>>(reference, target, transSpline, d_newX, d_newY, d_newZ, d_diff, tempPow);

  CUDAkernel_calculateSSDGradient_doCalculation<<< gridSpline, threadsSpline >>>(reference, target, transSpline, SSDGradient, d_newX, d_newY, d_newZ, d_diff, tempPow);
  //CUDAkernel_calculateSSDGradient_doCalculationY<<< gridSpline, threadsSpline >>>(reference, target, transSpline, SSDGradient, d_newX, d_newY, d_newZ, d_diff, tempPow);
  //CUDAkernel_calculateSSDGradient_doCalculationZ<<< gridSpline, threadsSpline >>>(reference, target, transSpline, SSDGradient, d_newX, d_newY, d_newZ, d_diff, tempPow);


  CUT_CHECK_ERROR("Kernel execution failed");
  /**/
  
  
  CUDA_SAFE_CALL( cudaMemcpy(buffer, SSDGradient, splineSizeX*splineSizeY*splineSizeZ*3*sizeof(float), cudaMemcpyDeviceToHost) );

  int i,j,k,n;
  float max=0;
  for(k=0;k<splineSizeZ;k++){
    for(j=0;j<splineSizeY;j++){
      for(i=0;i<splineSizeX;i++){
  for(n=0;n<3;n++){
    //printf("%d %d %d %d %lf \n", i,j,k,n,buffer[n*splineSizeX*splineSizeY*splineSizeZ+k*splineSizeX*splineSizeY+j*splineSizeX+i]);
    if(fabs(buffer[n*splineSizeX*splineSizeY*splineSizeZ+k*splineSizeX*splineSizeY+j*splineSizeX+i])>max)
      max=fabs(buffer[n*splineSizeX*splineSizeY*splineSizeZ+k*splineSizeX*splineSizeY+j*splineSizeX+i]);
  }
      }
    }
  }
  printf("max=%lf\n", max);

  free(buffer);
  /**/

  
  CUDA_SAFE_CALL( cudaFree(d_newX));
  CUDA_SAFE_CALL( cudaFree(d_newY));
  CUDA_SAFE_CALL( cudaFree(d_newZ));
  CUDA_SAFE_CALL( cudaFree(d_diff));
  

  return max;
}
