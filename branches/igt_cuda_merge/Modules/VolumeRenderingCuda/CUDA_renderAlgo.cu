extern "C" {
#include "CUDA_renderAlgo.h"
}

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

//cuda includes
#include "cudaTypeRange.h"
#include <cutil.h>

// vtk includes
//#include "vtkType.h"
// or use these defines. they work too.
#define VTK_CHAR            2
#define VTK_UNSIGNED_CHAR   3
#define VTK_SHORT           4
#define VTK_UNSIGNED_SHORT  5
#define VTK_INT             6
#define VTK_UNSIGNED_INT    7
#define VTK_FLOAT          10
#define VTK_DOUBLE         11


#define BLOCK_DIM2D 16// this must be set to 4 or more
#define SQR(X) ((X) * (X) )

template <typename T>
__device__ T interpolate(float posX, float posY, float posZ,
				     T val1, T val2, T val3, T val4, T val5, T val6, T val7,T val8)
{
  float revX= 1-posX;
  float revY= 1-posY;
  float revZ= 1-posZ;
  
  return ((T) (revX * revY * revZ * val1 +
	           revX * revY * posZ * val2 +
	           revX * posY * revZ * val3 +
	           revX * posY * posZ * val4 +
	           posX * revY * revZ * val5 +
	           posX * revY * posZ * val6 +
	           posX * posY * revZ * val7 +
	           posX * posY * posZ * val8)
	  );
}

template <typename T>
__global__ void CUDAkernel_renderAlgo_doIntegrationRender(
							  const cudaRendererInformation renInfo,
							  const cudaVolumeInformation volInfo
							  )
{
  int xIndex = blockDim.x *blockIdx.x + threadIdx.x;
  int yIndex = blockDim.y *blockIdx.y + threadIdx.y;

  __shared__ float2          s_minmaxTrace[BLOCK_DIM2D*BLOCK_DIM2D];      //starting and ending step of ray tracing 
  __shared__ float           s_rayMap[BLOCK_DIM2D*BLOCK_DIM2D*6];         //ray map: position and orientation of ray after translation and rotation transformation
  __shared__ float3          s_dsize;                                     //display size (x, y, dummy)
  __shared__ float3          s_vsize;                                     //voxel size, vtk spacing
  __shared__ float3          s_size;                                      //3D data size
  __shared__ float           s_minmax[6];                                 //region of interest of 3D data (minX, maxX, minY, maxY, minZ, maxZ)
  __shared__ float3          s_outputVal[BLOCK_DIM2D*BLOCK_DIM2D];        //output value
  __shared__ float           s_remainingOpacity[BLOCK_DIM2D*BLOCK_DIM2D]; //integration value of alpha
  __shared__ float           s_zBuffer[BLOCK_DIM2D*BLOCK_DIM2D];          // z buffer

  float test;

  int tempacc=threadIdx.x + threadIdx.y * BLOCK_DIM2D; //index in grid

  __syncthreads();	

  //copying variables into shared memory
  if(tempacc <3){ 
    s_dsize.x = renInfo.Resolution.x;
    s_dsize.y = renInfo.Resolution.y;
    s_vsize   = volInfo.Spacing;
    s_size.x  =  volInfo.VolumeSize.x;
    s_size.y  =  volInfo.VolumeSize.y;
    s_size.z  =  volInfo.VolumeSize.z;
  }else if(tempacc < 9){ 
    s_minmax[xIndex%6] = volInfo.MinMaxValue[xIndex%6];
  }

  __syncthreads();

  T typeMin = (T)volInfo.FunctionRange[0];
  T typeMax = (T)volInfo.FunctionRange[1];
  GetTypeRange<T>(typeMin, typeMax);  

  int outindex = xIndex + yIndex * s_dsize.x; // index of result image

  //initialization of variables in shared memory

  s_remainingOpacity[tempacc] = 1.0;
  s_outputVal[tempacc].x = 0;
  s_outputVal[tempacc].y = 0;
  s_outputVal[tempacc].z = 0;
  if(xIndex < s_dsize.x && yIndex < s_dsize.y){
    s_zBuffer[tempacc] = 10000;// (renInfo.ClippingRange.y * renInfo.ClippingRange.x / (renInfo.ClippingRange.x - renInfo.ClippingRange.y)) / (renInfo.ZBuffer[outindex] - renInfo.ClippingRange.y / (renInfo.ClippingRange.y - renInfo.ClippingRange.x));
  } else /* outside of screen */ {
    s_zBuffer[tempacc]=0;
  }
  __syncthreads();

  // lens map for perspective projection

  /*
    camera model start here
  */
  
  s_rayMap[tempacc*6]   = renInfo.CameraPos.x + s_size.x * s_vsize.x / 2.0f;
  s_rayMap[tempacc*6+1] = renInfo.CameraPos.y + s_size.y * s_vsize.y / 2.0f;
  s_rayMap[tempacc*6+2] = renInfo.CameraPos.z + s_size.z * s_vsize.z / 2.0f;
  

  float verX, verY, verZ;
  float horX, horY, horZ;
  
  float dot = renInfo.ViewUp.x * renInfo.CameraDirection.x +
              renInfo.ViewUp.y * renInfo.CameraDirection.y + 
              renInfo.ViewUp.z * renInfo.CameraDirection.z;

  verX = renInfo.ViewUp.x - dot * renInfo.CameraDirection.x;
  verY = renInfo.ViewUp.y - dot * renInfo.CameraDirection.y;
  verZ = renInfo.ViewUp.z - dot * renInfo.CameraDirection.z;

  horX=verY * renInfo.CameraDirection.z - verZ * renInfo.CameraDirection.y;
  horY=verZ * renInfo.CameraDirection.x - verX * renInfo.CameraDirection.z;
  horZ=verX * renInfo.CameraDirection.y - verY * renInfo.CameraDirection.x;

  float posHor= (xIndex-s_dsize.x*0.5) / s_dsize.x*0.27;
  float posVer= (yIndex-s_dsize.y*0.5) / s_dsize.x*0.27;
  
  s_rayMap[tempacc*6+3] = (renInfo.CameraDirection.x + posHor * horX + posVer * verX);
  s_rayMap[tempacc*6+4] = (renInfo.CameraDirection.y + posHor * horY + posVer * verY);
  s_rayMap[tempacc*6+5] = (renInfo.CameraDirection.z + posHor * horZ + posVer * verZ);

  /*
    camera model end here
  */
 
  //initialize variables for calculating starting and ending point of ray tracing

  s_minmaxTrace[tempacc].x = -100000.0f;
  s_minmaxTrace[tempacc].y = 100000.0f;

  __syncthreads();
  
  //normalize ray vector

  float getmax = fabs(s_rayMap[tempacc*6+3] / s_vsize.x);
  if(fabs(s_rayMap[tempacc*6+4] / s_vsize.y) > getmax) 
     getmax = fabs(s_rayMap[tempacc*6+4]/s_vsize.y);
  if(fabs(s_rayMap[tempacc*6+5] / s_vsize.z) > getmax) 
     getmax = fabs(s_rayMap[tempacc*6+5]/s_vsize.z);

  if(getmax!=0){
    float temp= 1.0f/getmax;
    s_rayMap[tempacc*6+3] *= temp;
    s_rayMap[tempacc*6+4] *= temp;
    s_rayMap[tempacc*6+5] *= temp;
  }

  float stepSize = sqrtf(s_rayMap[tempacc*6+3] * s_rayMap[tempacc*6+3] + 
                         s_rayMap[tempacc*6+4] * s_rayMap[tempacc*6+4] + 
                         s_rayMap[tempacc*6+5] * s_rayMap[tempacc*6+5]);
  __syncthreads();

  //calculating starting and ending point of ray tracing
 if(s_rayMap[tempacc*6+3] > 1.0e-3){
    s_minmaxTrace[tempacc].y = ( ((s_minmax[1]-2)*s_vsize.x - s_rayMap[tempacc*6]) / s_rayMap[tempacc*6+3] );
    s_minmaxTrace[tempacc].x = ( ((s_minmax[0]+2)*s_vsize.x - s_rayMap[tempacc*6]) / s_rayMap[tempacc*6+3] );
  }
  else if(s_rayMap[tempacc*6+3] < -1.0e-3){
    s_minmaxTrace[tempacc].x = ( ((s_minmax[1]-2)*s_vsize.x - s_rayMap[tempacc*6]) / s_rayMap[tempacc*6+3] );
    s_minmaxTrace[tempacc].y = ( ((s_minmax[0]+2)*s_vsize.x - s_rayMap[tempacc*6]) / s_rayMap[tempacc*6+3] );
  }
  
  if(s_rayMap[tempacc*6+4] > 1.0e-3){
    test = ( ((s_minmax[3]-2)*s_vsize.y - s_rayMap[tempacc*6+1]) / s_rayMap[tempacc*6+4] );
    if( test < s_minmaxTrace[tempacc].y){
      s_minmaxTrace[tempacc].y = test;
    }
    test = ( ((s_minmax[2]+2)*s_vsize.y - s_rayMap[tempacc*6+1]) / s_rayMap[tempacc*6+4] );
    if( test > s_minmaxTrace[tempacc].x){
      s_minmaxTrace[tempacc].x = test;
    }
  }
  else if(s_rayMap[tempacc*6+4] < -1.0e-3){
    test = ( ((s_minmax[3]-2)*s_vsize.y - s_rayMap[tempacc*6+1]) / s_rayMap[tempacc*6+4] );
    if( test > s_minmaxTrace[tempacc].x){
      s_minmaxTrace[tempacc].x = test;
    }
    test = ( ((s_minmax[2]+2)*s_vsize.y - s_rayMap[tempacc*6+1]) / s_rayMap[tempacc*6+4] );
    if( test < s_minmaxTrace[tempacc].y){
      s_minmaxTrace[tempacc].y = test;
    }
  }
  

  if(s_rayMap[tempacc*6+5] > 1.0e-3){
    test = ( ((s_minmax[5]-2)*s_vsize.z - s_rayMap[tempacc*6+2]) / s_rayMap[tempacc*6+5] );
    if( test < s_minmaxTrace[tempacc].y){
      s_minmaxTrace[tempacc].y = test;
    }
    test = ( ((s_minmax[4]+2)*s_vsize.z - s_rayMap[tempacc*6+2]) / s_rayMap[tempacc*6+5] );
    if( test > s_minmaxTrace[tempacc].x){
      s_minmaxTrace[tempacc].x = test;
    }
  }
  else if(s_rayMap[tempacc*6+5] < -1.0e-3){
    test = ( ((s_minmax[5]-2)*s_vsize.z - s_rayMap[tempacc*6+2]) / s_rayMap[tempacc*6+5] );
    if( test > s_minmaxTrace[tempacc].x){
      s_minmaxTrace[tempacc].x = test;
    }
    test = ( ((s_minmax[4]+2)*s_vsize.z - s_rayMap[tempacc*6+2]) / s_rayMap[tempacc*6+5] );
    if( test < s_minmaxTrace[tempacc].y){
      s_minmaxTrace[tempacc].y = test;
    }
  }
  __syncthreads();

  //ray tracing start from here

  float tempx,tempy,tempz; // variables to store current position
  float pos = 0; //current step distance from camera

  //float temp; //temporary variable to store data during calculation
  T tempValue;
  int tempIndex;
  float alpha; //alpha value of current voxel
  float initialZBuffer=s_zBuffer[tempacc]; //initial zBuffer from input

  //perform ray tracing until integration of alpha value reach threshold 
  
  while((s_minmaxTrace[tempacc].y - s_minmaxTrace[tempacc].x) >= pos) {
    
    //calculate current position in ray tracing

    tempx = ( s_rayMap[tempacc*6+0] + ((int)s_minmaxTrace[tempacc].x + pos) * s_rayMap[tempacc*6+3]);
    tempy = ( s_rayMap[tempacc*6+1] + ((int)s_minmaxTrace[tempacc].x + pos) * s_rayMap[tempacc*6+4]);
    tempz = ( s_rayMap[tempacc*6+2] + ((int)s_minmaxTrace[tempacc].x + pos) * s_rayMap[tempacc*6+5]);
    
    tempx /= s_vsize.x;
    tempy /= s_vsize.y;
    tempz /= s_vsize.z;
    
    // if current position is in ROI
    if(tempx >= s_minmax[0] && tempx < s_minmax[1] &&
       tempy >= s_minmax[2] && tempy < s_minmax[3] &&
       tempz >= s_minmax[4] && tempz < s_minmax[5] && 
       pos + s_minmaxTrace[tempacc].x >= -500 /*renInfo.ClippingRange[0]*/)
       {
      //check whether current position is in front of z buffer wall
      if((pos + s_minmaxTrace[tempacc].x)*stepSize < initialZBuffer)
      { 

	tempValue=((T*)volInfo.SourceData)[(int)(__float2int_rn(tempz)*s_size.x*s_size.y + 
                                             __float2int_rn(tempy)*s_size.x +
                                             __float2int_rn(tempx))];
	/*interpolation start here*/
	float posX = tempx-__float2int_rd(tempx);
	float posY = tempy-__float2int_rd(tempy);
	float posZ = tempz-__float2int_rd(tempz);

	/*
	tempValue=interpolate((float)0,(float)0,(float)0,
			      ((T*)volInfo.SourceData)[(int)((int)(tempz)*s_size.x*s_size.y + 
			                                     (int)(tempy)*s_size.x + 
			                                     (int)(tempx))],
			                                     (T)0,(T)0,(T)0,(T)0,(T)0,(T)0,(T)0);
	*/      
	int base = __float2int_rd((tempz))*s_size.x*s_size.y + 
	           __float2int_rd((tempy))*s_size.x + 
	           __float2int_rd((tempx));
	
	tempValue=interpolate(posX, posY,0.0,
			      ((T*)volInfo.SourceData)[base],
			      ((T*)volInfo.SourceData)[(int)(base + s_size.x*s_size.y)],
			      ((T*)volInfo.SourceData)[(int)(base + s_size.x)],
			      ((T*)volInfo.SourceData)[(int)(base + s_size.x*s_size.y + s_size.x)],
			      ((T*)volInfo.SourceData)[(int)(base + 1)],
			      ((T*)volInfo.SourceData)[(int)(base + s_size.x*s_size.y + 1)],
			      ((T*)volInfo.SourceData)[(int)(base + s_size.x + 1)],
			      ((T*)volInfo.SourceData)[(int)(base + s_size.x*s_size.y + s_size.x + 1)]);
	/*interpolation end here*/

	if( tempValue >=(T)volInfo.MinThreshold && tempValue <= (T)volInfo.MaxThreshold){ 
	  
	  tempIndex=__float2int_rn((volInfo.FunctionSize-1)*(float)(tempValue-typeMin)/(float)(typeMax-typeMin));
	  alpha=volInfo.AlphaTransferFunction[tempIndex];
	  
	  if(s_zBuffer[tempacc] > (pos + s_minmaxTrace[tempacc].x) * stepSize)
	  {
	    s_zBuffer[tempacc] = (pos + s_minmaxTrace[tempacc].x) * stepSize;
	  }
	  if(s_remainingOpacity[tempacc] > 0.02){ // check if remaining opacity has reached threshold(0.02)
	    s_outputVal[tempacc].x += s_remainingOpacity[tempacc] * alpha * volInfo.ColorTransferFunction[tempIndex*3];
	    s_outputVal[tempacc].y += s_remainingOpacity[tempacc] * alpha * volInfo.ColorTransferFunction[tempIndex*3+1];
	    s_outputVal[tempacc].z += s_remainingOpacity[tempacc] * alpha * volInfo.ColorTransferFunction[tempIndex*3+2];
	    s_remainingOpacity[tempacc] *= (1.0 - alpha);
	  }else{
	    pos = s_minmaxTrace[tempacc].y - s_minmaxTrace[tempacc].x;
	  }
	}
	

      } else { // current position is behind z buffer wall
	if(xIndex < s_dsize.x && yIndex < s_dsize.y){
	  
	  s_outputVal[tempacc].x += s_remainingOpacity[tempacc] * renInfo.OutputImage[outindex].x;
	  s_outputVal[tempacc].y += s_remainingOpacity[tempacc] * renInfo.OutputImage[outindex].y;
	  s_outputVal[tempacc].z += s_remainingOpacity[tempacc] * renInfo.OutputImage[outindex].z;
	  
	}
	  pos = s_minmaxTrace[tempacc].y - s_minmaxTrace[tempacc].x;
      }
    }
    pos += volInfo.SteppingSize;
  }

  //write to output

  if(xIndex < s_dsize.x && yIndex < s_dsize.y){
    renInfo.OutputImage[outindex]=make_uchar4(s_outputVal[tempacc].x * 255.0, 
					                          s_outputVal[tempacc].y * 255.0, 
					                          s_outputVal[tempacc].z * 255.0, 
					                         (1 - s_remainingOpacity[tempacc])*255.0);
    renInfo.ZBuffer[outindex]=s_zBuffer[tempacc];
  }
}

extern "C"
void CUDArenderAlgo_doRender(const cudaRendererInformation& rendererInfo,
							 const cudaVolumeInformation& volumeInfo)
{
  int blockX=((rendererInfo.Resolution.x-1)/ BLOCK_DIM2D) + 1;
  int blockY=((rendererInfo.Resolution.y-1)/ BLOCK_DIM2D) + 1;

  // setup execution parameters

  dim3 grid(blockX, blockY, 1);
  dim3 threads(BLOCK_DIM2D, BLOCK_DIM2D, 1);

  CUT_DEVICE_INIT();
  
// The CUDA Kernel Function Definition, so we do not have to write it down below
#define CUDA_KERNEL_CALL(ID, TYPE)   \
	if (volumeInfo.InputDataType == ID) \
	 CUDAkernel_renderAlgo_doIntegrationRender<TYPE> <<< grid, threads >>>( \
	 rendererInfo, \
	 volumeInfo)

// Add all the other types.
  CUDA_KERNEL_CALL(VTK_UNSIGNED_CHAR, unsigned char);
  else CUDA_KERNEL_CALL(VTK_CHAR, char);
  else CUDA_KERNEL_CALL(VTK_SHORT, short);
  else CUDA_KERNEL_CALL(VTK_UNSIGNED_SHORT, unsigned short);
  else CUDA_KERNEL_CALL(VTK_FLOAT, float);
  else CUDA_KERNEL_CALL(VTK_DOUBLE, double);
  else CUDA_KERNEL_CALL(VTK_INT, int);


  CUT_CHECK_ERROR("Kernel execution failed");

  return;
}
