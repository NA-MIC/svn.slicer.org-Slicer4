// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil.h>

// includes, kernels
extern "C" {
#include "CUDA_renderAlgo.h"
}

#define BLOCK_DIM2D 4 // this must be set to 4 or more
#define SQR(X) ((X) * (X) )

__constant__ float c_renderAlgo_size[3];
__constant__ float c_renderAlgo_dsize[2];
__constant__ float c_renderAlgo_minmax[6];
__constant__ float c_renderAlgo_lightVec[3];

__constant__ float c_renderAlgo_rotationMatrix1[4];
__constant__ float c_renderAlgo_rotationMatrix2[4];
__constant__ float c_renderAlgo_rotationMatrix3[4];
__constant__ float c_renderAlgo_vsize[3];
__constant__ float c_renderAlgo_disp[3];

// Do hybrid rendering (combination between ray tracing volume rendering and MIP rendering). Here, rendering parameter was set to (1.0-transparencyLevel) MIP and (transparencyLevel) ray tracing volume rendering. Transparency level 1 means fully oblique, and transparency level 0 means fully transparent.

template <typename T>
__global__ void CUDAkernel_renderAlgo_doHybridRender(T* d_sourceData, 
													 float* colorTransferFunction,
													 float* alphaTransferFunction,
                                                     unsigned char minThreshold, 
                                                     unsigned char maxThreshold, 
                                                     int sliceDistance, 
                                                     float transparencyLevel, 
                                                     uchar4* d_resultImage)
{
  int xIndex = blockDim.x *blockIdx.x + threadIdx.x;
  int yIndex = blockDim.y *blockIdx.y + threadIdx.y;

  __shared__ float2 s_minmaxTrace[BLOCK_DIM2D*BLOCK_DIM2D];
  __shared__ uchar4 s_resultImage[BLOCK_DIM2D*BLOCK_DIM2D];
  __shared__ float s_rayMap[BLOCK_DIM2D*BLOCK_DIM2D*6];
  __shared__ float s_rotationMatrix1[4];
  __shared__ float s_rotationMatrix2[4];
  __shared__ float s_rotationMatrix3[4];
  __shared__ float s_dsize[3];
  __shared__ float s_vsize[3];
  __shared__ float s_size[3];
  __shared__ float s_disp[3];
  __shared__ float s_lightVec[3];
  __shared__ float s_minmax[6];
  __shared__ float s_maxVal[BLOCK_DIM2D*BLOCK_DIM2D];
  __shared__ int s_pos[BLOCK_DIM2D*BLOCK_DIM2D];
  float4 s_shadeField;

  float test;
  float tempf;
  float posX, posY;
  
  int tempacc=threadIdx.x+threadIdx.y*BLOCK_DIM2D;

  __syncthreads();

  if(tempacc <4){ 
    s_rotationMatrix1[xIndex%4]=c_renderAlgo_rotationMatrix1[xIndex%4];
    s_rotationMatrix2[xIndex%4]=c_renderAlgo_rotationMatrix2[xIndex%4];
    s_rotationMatrix3[xIndex%4]=c_renderAlgo_rotationMatrix3[xIndex%4];
  }else if(tempacc < 7){ 
    s_dsize[xIndex%2]=c_renderAlgo_dsize[xIndex%2];
    s_vsize[xIndex%3]=c_renderAlgo_vsize[xIndex%3];
    s_size[xIndex%3]=c_renderAlgo_size[xIndex%3];
    s_disp[xIndex%3]=c_renderAlgo_disp[xIndex%3];
    s_lightVec[xIndex%3]=c_renderAlgo_lightVec[xIndex%3];
  }else if(tempacc < 13){ 
    s_minmax[xIndex%6]=c_renderAlgo_minmax[xIndex%6];
  }

  s_maxVal[tempacc]=0;
  s_pos[tempacc]=-1;
  s_resultImage[tempacc]=make_uchar4(0,0,0,0);

  __syncthreads();

  posX=(xIndex-s_dsize[0]*0.5)/2.0;
  posY=(yIndex-s_dsize[1]*0.5)/2.0;

  s_rayMap[tempacc*6+3] = (s_rotationMatrix1[2])*(float)s_vsize[0];
  s_rayMap[tempacc*6+4] = (s_rotationMatrix2[2])*(float)s_vsize[1];
  s_rayMap[tempacc*6+5] = (s_rotationMatrix3[2])*(float)s_vsize[2];

  s_rayMap[tempacc*6] = ((float)s_size[0]/2.0f + ((posX+s_rotationMatrix1[3])*s_rotationMatrix1[0] + (posY+s_rotationMatrix2[3])*s_rotationMatrix1[1] + (s_disp[2])*s_rotationMatrix1[2])*s_vsize[0]);
  s_rayMap[tempacc*6+1] = ((float)s_size[1]/2.0f + ((posX+s_rotationMatrix1[3])*s_rotationMatrix2[0] + (posY+s_rotationMatrix2[3])*s_rotationMatrix2[1] + (s_disp[2])*s_rotationMatrix2[2])*s_vsize[1]);
  s_rayMap[tempacc*6+2] = ((float)s_size[2]/2.0f + ((posX+s_rotationMatrix1[3])*s_rotationMatrix3[0] + (posY+s_rotationMatrix2[3])*s_rotationMatrix3[1] + (s_disp[2])*s_rotationMatrix3[2])*s_vsize[2]);  
  
  s_minmaxTrace[tempacc].x=-100000.0f;
  s_minmaxTrace[tempacc].y=100000.0f;

  __syncthreads();

  float getmax = fabs(s_rayMap[tempacc*6+3]);
  if(fabs(s_rayMap[tempacc*6+4])>getmax) getmax = fabs(s_rayMap[tempacc*6+4]);
  if(fabs(s_rayMap[tempacc*6+5])>getmax) getmax = fabs(s_rayMap[tempacc*6+5]);

  if(getmax!=0){
    float temp= 1.0f/getmax;
    s_rayMap[tempacc*6+3]*=temp;
    s_rayMap[tempacc*6+4]*=temp;
    s_rayMap[tempacc*6+5]*=temp;
  }

  __syncthreads();

  if(s_rayMap[tempacc*6+3] > 1.0e-3){
    s_minmaxTrace[tempacc].y = ( (s_minmax[1]-2-s_rayMap[tempacc*6])/s_rayMap[tempacc*6+3] );
    s_minmaxTrace[tempacc].x = ( (s_minmax[0]+2-s_rayMap[tempacc*6])/s_rayMap[tempacc*6+3] );
  }
  else if(s_rayMap[tempacc*6+3] < -1.0e-3){
    s_minmaxTrace[tempacc].x = ( (s_minmax[1]-2-s_rayMap[tempacc*6])/s_rayMap[tempacc*6+3] );
    s_minmaxTrace[tempacc].y = ( (s_minmax[0]+2-s_rayMap[tempacc*6])/s_rayMap[tempacc*6+3] );
  }
  
  if(s_rayMap[tempacc*6+4] > 1.0e-3){
    test = ( (s_minmax[3]-2-s_rayMap[tempacc*6+1])/s_rayMap[tempacc*6+4] );
    if( test < s_minmaxTrace[tempacc].y){
      s_minmaxTrace[tempacc].y = test;
    }
    test = ( (s_minmax[2]+2-s_rayMap[tempacc*6+1])/s_rayMap[tempacc*6+4] );
    if( test > s_minmaxTrace[tempacc].x){
      s_minmaxTrace[tempacc].x = test;
    }
  }
  else if(s_rayMap[tempacc*6+4] < -1.0e-3){
    test = ( (s_minmax[3]-2-s_rayMap[tempacc*6+1])/s_rayMap[tempacc*6+4] );
    if( test > s_minmaxTrace[tempacc].x){
      s_minmaxTrace[tempacc].x = test;
    }
    test = ( (s_minmax[2]+2-s_rayMap[tempacc*6+1])/s_rayMap[tempacc*6+4] );
    if( test < s_minmaxTrace[tempacc].y){
      s_minmaxTrace[tempacc].y = test;
    }
  }
  

  if(s_rayMap[tempacc*6+5] > 1.0e-3){
    test = ( (s_minmax[5]-2-s_rayMap[tempacc*6+2])/s_rayMap[tempacc*6+5] );
    if( test < s_minmaxTrace[tempacc].y){
      s_minmaxTrace[tempacc].y = test;
    }
    test = ( (s_minmax[4]+2-s_rayMap[tempacc*6+2])/s_rayMap[tempacc*6+5] );
    if( test > s_minmaxTrace[tempacc].x){
      s_minmaxTrace[tempacc].x = test;
    }
  }
  else if(s_rayMap[tempacc*6+5] < -1.0e-3){
    test = ( (s_minmax[5]-2-s_rayMap[tempacc*6+2])/s_rayMap[tempacc*6+5] );
    if( test > s_minmaxTrace[tempacc].x){
      s_minmaxTrace[tempacc].x = test;
    }
    test = ( (s_minmax[4]+2-s_rayMap[tempacc*6+2])/s_rayMap[tempacc*6+5] );
    if( test < s_minmaxTrace[tempacc].y){
      s_minmaxTrace[tempacc].y = test;
    }
  }
  __syncthreads();

  float val=0;
  float tempData=0;
  int tempx,tempy,tempz;
  int pos=0;

  float temp;

  while((s_minmaxTrace[tempacc].y-s_minmaxTrace[tempacc].x)>=pos){
    
    tempx = (int)( s_rayMap[tempacc*6+0]+((int)s_minmaxTrace[tempacc].x+pos)*s_rayMap[tempacc*6+3]);
    tempy = (int)( s_rayMap[tempacc*6+1]+((int)s_minmaxTrace[tempacc].x+pos)*s_rayMap[tempacc*6+4]);
    tempz = (int)( s_rayMap[tempacc*6+2]+((int)s_minmaxTrace[tempacc].x+pos)*s_rayMap[tempacc*6+5]);
    
    if(tempx >= s_minmax[0] && tempx <= s_minmax[1] && tempy >= s_minmax[2] && tempy <= s_minmax[3] && tempz >= s_minmax[4] && tempz <= s_minmax[5] && pos+s_minmaxTrace[tempacc].x >=sliceDistance){

      temp=d_sourceData[(int)(tempz*s_size[0]*s_size[1]+tempy*s_size[0]+tempx)];

      if(temp>s_maxVal[tempacc])s_maxVal[tempacc]=temp;

      if(s_pos[tempacc]==-1 && temp >=minThreshold && temp <= maxThreshold  && pos+s_minmaxTrace[tempacc].x >=sliceDistance){
  s_pos[tempacc]=pos;
      }
                  
    }
    pos++;
    
  }
  
  pos=s_pos[tempacc];
  
  tempx = (int)( s_rayMap[tempacc*6+0]+((int)s_minmaxTrace[tempacc].x+pos)*s_rayMap[tempacc*6+3]);
  tempy = (int)( s_rayMap[tempacc*6+1]+((int)s_minmaxTrace[tempacc].x+pos)*s_rayMap[tempacc*6+4]);
  tempz = (int)( s_rayMap[tempacc*6+2]+((int)s_minmaxTrace[tempacc].x+pos)*s_rayMap[tempacc*6+5]);
  
  if(s_pos[tempacc]!=-1 ){
    
    if(__float2int_rd(pos+s_minmaxTrace[tempacc].x) == sliceDistance)val=d_sourceData[(int)(tempz*s_size[0]*s_size[1]+tempy*s_size[0]+tempx)];
    
    tempData=d_sourceData[(int)(tempz*s_size[0]*s_size[1]+tempy*s_size[0]+tempx)];

    s_shadeField.x = ((float)d_sourceData[(int)(tempz*s_size[0]*s_size[1]+tempy*s_size[0]+tempx+1)]-(float)d_sourceData[(int)(tempz*s_size[0]*s_size[1]+tempy*s_size[0]+tempx-1)]);
    s_shadeField.y = ((float)d_sourceData[(int)(tempz*s_size[0]*s_size[1]+(tempy+1)*s_size[0]+tempx)]-(float)d_sourceData[(int)(tempz*s_size[0]*s_size[1]+(tempy-1)*s_size[0]+tempx)]);
    s_shadeField.z = ((float)d_sourceData[(int)((tempz+1)*s_size[0]*s_size[1]+tempy*s_size[0]+tempx)]-(float)d_sourceData[(int)((tempz-1)*s_size[0]*s_size[1]+tempy*s_size[0]+tempx)]);
    
    
    tempf = 1.0/sqrt(SQR(s_shadeField.x) + SQR(s_shadeField.y) + SQR(s_shadeField.z));
    s_shadeField.x = tempf * s_shadeField.x;
    s_shadeField.y = tempf * s_shadeField.y;
    s_shadeField.z = tempf * s_shadeField.z;
    
    if(val==0.0){
      val = (s_shadeField.x*s_lightVec[0]+s_shadeField.y*s_lightVec[1]+s_shadeField.z*s_lightVec[2]);
    }
    
    if(val<0.0)val=0.0;
    
    if(val<=1.0){
      
      // Set rendering parameter here: (0.2 MIP rendering and 0.8 ray casting volume rendering).

      val=(transparencyLevel*val+(1.0-transparencyLevel)*s_maxVal[tempacc]/255.0);
      s_resultImage[tempacc]=make_uchar4((unsigned char)( colorTransferFunction[(int)(tempData*3)]*val * 256) ,
					 (unsigned char)( colorTransferFunction[(int)(tempData*3+1)]*val *256), 
					 (unsigned char)( colorTransferFunction[(int)(tempData*3+2)]*val * 256), 
					 (unsigned char) 255);
      }else{
      s_resultImage[tempacc]=make_uchar4((unsigned char)val, (unsigned char)val, (unsigned char)val, 255 );
    }
    
  }
  
  //__syncthreads();

  d_resultImage[(int)(xIndex+yIndex*c_renderAlgo_dsize[0])]=s_resultImage[tempacc];

}

void CUDArenderAlgo_doRender(uchar4* outputData,
                             void* renderData,
                             int inputDataType,
                             float* rotationMatrix, 
                             float* colorTransferFunction,
                             float* alphaTransferFunction,
                             float* minmax, float* lightVec, 
                             int sizeX, int sizeY, int sizeZ, 
                             int dsizeX, int dsizeY, 
                             float dispX, float dispY, float dispZ, 
                             float voxelSizeX, float voxelSizeY, float voxelSizeZ, 
                             int minThreshold, int maxThreshold, 
                             int sliceDistance)
{
  float size[3]={sizeX, sizeY, sizeZ};
  float dsize[2]={dsizeX, dsizeY};

  CUDA_SAFE_CALL( cudaMemcpyToSymbol(c_renderAlgo_size, size, sizeof(float)*3, 0));
  CUDA_SAFE_CALL( cudaMemcpyToSymbol(c_renderAlgo_dsize, dsize, sizeof(float)*2, 0));


  // setup execution parameters

  dim3 grid(dsizeX / BLOCK_DIM2D, dsizeY / BLOCK_DIM2D, 1);
  dim3 threads(BLOCK_DIM2D, BLOCK_DIM2D, 1);

  CUT_DEVICE_INIT();

  // allocate host memory

  float h_size[3]={sizeX, sizeY, sizeZ};
  float h_dsize[2]={dsizeX, dsizeY};
  float h_vsize[3]={voxelSizeX, voxelSizeY, voxelSizeZ};
  float h_disp[3]={dispX, dispY, dispZ};
  
  float h_minmax[6]={minmax[0], minmax[1], minmax[2], minmax[3], minmax[4], minmax[5]};
  float h_lightVec[3]={lightVec[0], lightVec[1], lightVec[2]};

  // copy host memory to device
  
  CUDA_SAFE_CALL( cudaMemcpyToSymbol(c_renderAlgo_size, h_size, sizeof(float)*3, 0));
  CUDA_SAFE_CALL( cudaMemcpyToSymbol(c_renderAlgo_dsize, h_dsize, sizeof(float)*2, 0));
  CUDA_SAFE_CALL( cudaMemcpyToSymbol(c_renderAlgo_vsize, h_vsize, sizeof(float)*3, 0));
  CUDA_SAFE_CALL( cudaMemcpyToSymbol(c_renderAlgo_disp, h_disp, sizeof(float)*3, 0));

  CUDA_SAFE_CALL( cudaMemcpyToSymbol(c_renderAlgo_rotationMatrix1, rotationMatrix, sizeof(float)*4, 0));
  CUDA_SAFE_CALL( cudaMemcpyToSymbol(c_renderAlgo_rotationMatrix2, rotationMatrix+4, sizeof(float)*4, 0));
  CUDA_SAFE_CALL( cudaMemcpyToSymbol(c_renderAlgo_rotationMatrix3, rotationMatrix+8, sizeof(float)*4, 0));
    
  CUDA_SAFE_CALL( cudaMemcpyToSymbol(c_renderAlgo_minmax, h_minmax, sizeof(float)*6, 0));
  CUDA_SAFE_CALL( cudaMemcpyToSymbol(c_renderAlgo_lightVec, h_lightVec, sizeof(float)*3, 0));

  // execute the kernel

  // Switch to various rendering methods.

  float transparencyLevel = 1.0;

#define CUDA_KERNEL_CALL(ID, TYPE)   \
	if (inputDataType == ID) \
	 CUDAkernel_renderAlgo_doHybridRender<<< grid, threads >>>( \
	 (TYPE*)renderData, \
	 colorTransferFunction, \
	 alphaTransferFunction, \
	 minThreshold, maxThreshold, \
	 sliceDistance, \
	 transparencyLevel, \
	 outputData)

  CUDA_KERNEL_CALL(1, unsigned char);
  
  CUT_CHECK_ERROR("Kernel execution failed");

  return;
}
