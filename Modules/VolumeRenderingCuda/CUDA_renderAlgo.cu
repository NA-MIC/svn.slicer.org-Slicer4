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

__device__ void CUDAkernel_SetRayMap(const int3& index, float* raymap, const cudaRendererInformation& renInfo, const cudaVolumeInformation& volInfo)
{
  float posHor= (index.x - renInfo.Resolution.y*0.5) / renInfo.Resolution.x*0.27;
  float posVer= (index.y - renInfo.Resolution.y*0.5) / renInfo.Resolution.x*0.27;
  
  raymap[index.z*6]   = renInfo.CameraPos.x + volInfo.VolumeSize.x * volInfo.Spacing.x / 2.0f;
  raymap[index.z*6+1] = renInfo.CameraPos.y + volInfo.VolumeSize.y * volInfo.Spacing.y / 2.0f;
  raymap[index.z*6+2] = renInfo.CameraPos.z + volInfo.VolumeSize.z * volInfo.Spacing.z / 2.0f;
  raymap[index.z*6+3] = (renInfo.CameraDirection.x + posHor * renInfo.HorizontalVec.x + posVer * renInfo.VerticalVec.x);
  raymap[index.z*6+4] = (renInfo.CameraDirection.y + posHor * renInfo.HorizontalVec.y + posVer * renInfo.VerticalVec.y);
  raymap[index.z*6+5] = (renInfo.CameraDirection.z + posHor * renInfo.HorizontalVec.z + posVer * renInfo.VerticalVec.z);
}

__device__ void CUDAkernel_CalculateRayEnds(const int3& index, float* minmax/*[6]*/, float2* minmaxTrace, float* rayMap, const float3& voxelSize)
{
 float test;
  //calculating starting and ending point of ray tracing
 if(rayMap[index.z*6+3] > 1.0e-3){
    minmaxTrace[index.z].y = ( ((minmax[1]-2)*voxelSize.x - rayMap[index.z*6]) / rayMap[index.z*6+3] );
    minmaxTrace[index.z].x = ( ((minmax[0]+2)*voxelSize.x - rayMap[index.z*6]) / rayMap[index.z*6+3] );
  }
  else if(rayMap[index.z*6+3] < -1.0e-3){
    minmaxTrace[index.z].x = ( ((minmax[1]-2)*voxelSize.x - rayMap[index.z*6]) / rayMap[index.z*6+3] );
    minmaxTrace[index.z].y = ( ((minmax[0]+2)*voxelSize.x - rayMap[index.z*6]) / rayMap[index.z*6+3] );
  }
  
  if(rayMap[index.z*6+4] > 1.0e-3){
    test = ( ((minmax[3]-2)*voxelSize.y - rayMap[index.z*6+1]) / rayMap[index.z*6+4] );
    if( test < minmaxTrace[index.z].y){
      minmaxTrace[index.z].y = test;
    }
    test = ( ((minmax[2]+2)*voxelSize.y - rayMap[index.z*6+1]) / rayMap[index.z*6+4] );
    if( test > minmaxTrace[index.z].x){
      minmaxTrace[index.z].x = test;
    }
  }
  else if(rayMap[index.z*6+4] < -1.0e-3){
    test = ( ((minmax[3]-2)*voxelSize.y - rayMap[index.z*6+1]) / rayMap[index.z*6+4] );
    if( test > minmaxTrace[index.z].x){
      minmaxTrace[index.z].x = test;
    }
    test = ( ((minmax[2]+2)*voxelSize.y - rayMap[index.z*6+1]) / rayMap[index.z*6+4] );
    if( test < minmaxTrace[index.z].y){
      minmaxTrace[index.z].y = test;
    }
  }
  

  if(rayMap[index.z*6+5] > 1.0e-3){
    test = ( ((minmax[5]-2)*voxelSize.z - rayMap[index.z*6+2]) / rayMap[index.z*6+5] );
    if( test < minmaxTrace[index.z].y){
      minmaxTrace[index.z].y = test;
    }
    test = ( ((minmax[4]+2)*voxelSize.z - rayMap[index.z*6+2]) / rayMap[index.z*6+5] );
    if( test > minmaxTrace[index.z].x){
      minmaxTrace[index.z].x = test;
    }
  }
  else if(rayMap[index.z*6+5] < -1.0e-3){
    test = ( ((minmax[5]-2)*voxelSize.z - rayMap[index.z*6+2]) / rayMap[index.z*6+5] );
    if( test > minmaxTrace[index.z].x){
      minmaxTrace[index.z].x = test;
    }
    test = ( ((minmax[4]+2)*voxelSize.z - rayMap[index.z*6+2]) / rayMap[index.z*6+5] );
    if( test < minmaxTrace[index.z].y){
      minmaxTrace[index.z].y = test;
    }
  }
}


template <typename T>
__global__ void CUDAkernel_renderAlgo_doIntegrationRender(
							  const cudaRendererInformation renInfo,
							  const cudaVolumeInformation volInfo
							  )
{
  int3 index;
  index.x = blockDim.x *blockIdx.x + threadIdx.x;
  index.y = blockDim.y *blockIdx.y + threadIdx.y;
  index.z = threadIdx.x + threadIdx.y * BLOCK_DIM2D; //index in grid

  __shared__ float2          s_minmaxTrace[BLOCK_DIM2D*BLOCK_DIM2D];      //starting and ending step of ray tracing 
  __shared__ float           s_rayMap[BLOCK_DIM2D*BLOCK_DIM2D*6];         //ray map: position and orientation of ray after translation and rotation transformation
  __shared__ float3          s_dsize;                                     //display size (x, y, dummy)
  __shared__ float3          s_vsize;                                     //voxel size, vtk spacing
  __shared__ float3          s_size;                                      //3D data size
  __shared__ float           s_minmax[6];                                 //region of interest of 3D data (minX, maxX, minY, maxY, minZ, maxZ)
  __shared__ float3          s_outputVal[BLOCK_DIM2D*BLOCK_DIM2D];        //output value
  __shared__ float           s_remainingOpacity[BLOCK_DIM2D*BLOCK_DIM2D]; //integration value of alpha
  __shared__ float           s_zBuffer[BLOCK_DIM2D*BLOCK_DIM2D];          // z buffer



  __syncthreads();	

  //copying variables into shared memory
  if(index.z < 3){ 
    s_dsize.x = renInfo.Resolution.x;
    s_dsize.y = renInfo.Resolution.y;
    s_vsize   = volInfo.Spacing;
    s_size.x  = volInfo.VolumeSize.x;
    s_size.y  = volInfo.VolumeSize.y;
    s_size.z  = volInfo.VolumeSize.z;
  }else if(index.z < 9){ 
    s_minmax[index.x%6] = volInfo.MinMaxValue[index.x%6];
  }

  __syncthreads();

  T typeMin = (T)volInfo.FunctionRange[0];
  T typeMax = (T)volInfo.FunctionRange[1];
  //GetTypeRange<T>(typeMin, typeMax);  

  int outindex = index.x + index.y * s_dsize.x; // index of result image

  //initialization of variables in shared memory

  s_remainingOpacity[index.z] = 1.0;
  s_outputVal[index.z].x = 0;
  s_outputVal[index.z].y = 0;
  s_outputVal[index.z].z = 0;
  if(index.x < s_dsize.x && index.y < s_dsize.y){
    s_zBuffer[index.z] = 10000;// (renInfo.ClippingRange.y * renInfo.ClippingRange.x / (renInfo.ClippingRange.x - renInfo.ClippingRange.y)) / (renInfo.ZBuffer[outindex] - renInfo.ClippingRange.y / (renInfo.ClippingRange.y - renInfo.ClippingRange.x));
  } else /* outside of screen */ {
    s_zBuffer[index.z]=0;
  }
  __syncthreads();

  // lens map for perspective projection
  CUDAkernel_SetRayMap(index, s_rayMap, renInfo, volInfo);
 
  //initialize variables for calculating starting and ending point of ray tracing

  s_minmaxTrace[index.z].x = -100000.0f;
  s_minmaxTrace[index.z].y = 100000.0f;

  __syncthreads();
  
  //normalize ray vector

  float getmax = fabs(s_rayMap[index.z*6+3] / s_vsize.x);
  if(fabs(s_rayMap[index.z*6+4] / s_vsize.y) > getmax) 
     getmax = fabs(s_rayMap[index.z*6+4]/s_vsize.y);
  if(fabs(s_rayMap[index.z*6+5] / s_vsize.z) > getmax) 
     getmax = fabs(s_rayMap[index.z*6+5]/s_vsize.z);

  if(getmax!=0){
    float temp= 1.0f/getmax;
    s_rayMap[index.z*6+3] *= temp;
    s_rayMap[index.z*6+4] *= temp;
    s_rayMap[index.z*6+5] *= temp;
  }

  float stepSize = sqrtf(s_rayMap[index.z*6+3] * s_rayMap[index.z*6+3] + 
                         s_rayMap[index.z*6+4] * s_rayMap[index.z*6+4] + 
                         s_rayMap[index.z*6+5] * s_rayMap[index.z*6+5]);
  __syncthreads();
  CUDAkernel_CalculateRayEnds(index, s_minmax, s_minmaxTrace, s_rayMap, volInfo.Spacing);
  __syncthreads();

  //ray tracing start from here

  float tempx,tempy,tempz; // variables to store current position
  float pos = 0; //current step distance from camera

  //float temp; //temporary variable to store data during calculation
  T tempValue;
  int tempIndex;
  float alpha; //alpha value of current voxel
  float initialZBuffer=s_zBuffer[index.z]; //initial zBuffer from input

  //perform ray tracing until integration of alpha value reach threshold 
  
  while((s_minmaxTrace[index.z].y - s_minmaxTrace[index.z].x) >= pos) {
    
    //calculate current position in ray tracing

    tempx = ( s_rayMap[index.z*6+0] + ((int)s_minmaxTrace[index.z].x + pos) * s_rayMap[index.z*6+3]);
    tempy = ( s_rayMap[index.z*6+1] + ((int)s_minmaxTrace[index.z].x + pos) * s_rayMap[index.z*6+4]);
    tempz = ( s_rayMap[index.z*6+2] + ((int)s_minmaxTrace[index.z].x + pos) * s_rayMap[index.z*6+5]);
    
    tempx /= s_vsize.x;
    tempy /= s_vsize.y;
    tempz /= s_vsize.z;
    
    // if current position is in ROI
    if(tempx >= s_minmax[0] && tempx < s_minmax[1] &&
       tempy >= s_minmax[2] && tempy < s_minmax[3] &&
       tempz >= s_minmax[4] && tempz < s_minmax[5] && 
       pos + s_minmaxTrace[index.z].x >= -500 /*renInfo.ClippingRange[0]*/)
       {
      //check whether current position is in front of z buffer wall
      if((pos + s_minmaxTrace[index.z].x)*stepSize < initialZBuffer)
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
	  
	  if(s_zBuffer[index.z] > (pos + s_minmaxTrace[index.z].x) * stepSize)
	  {
	    s_zBuffer[index.z] = (pos + s_minmaxTrace[index.z].x) * stepSize;
	  }
	  if(s_remainingOpacity[index.z] > 0.02){ // check if remaining opacity has reached threshold(0.02)
	    s_outputVal[index.z].x += s_remainingOpacity[index.z] * alpha * volInfo.ColorTransferFunction[tempIndex*3];
	    s_outputVal[index.z].y += s_remainingOpacity[index.z] * alpha * volInfo.ColorTransferFunction[tempIndex*3+1];
	    s_outputVal[index.z].z += s_remainingOpacity[index.z] * alpha * volInfo.ColorTransferFunction[tempIndex*3+2];
	    s_remainingOpacity[index.z] *= (1.0 - alpha);
	  }else{
	    pos = s_minmaxTrace[index.z].y - s_minmaxTrace[index.z].x;
	  }
	}
	

      } else { // current position is behind z buffer wall
	if(index.x < s_dsize.x && index.y < s_dsize.y){
	  
	  s_outputVal[index.z].x += s_remainingOpacity[index.z] * renInfo.OutputImage[outindex].x;
	  s_outputVal[index.z].y += s_remainingOpacity[index.z] * renInfo.OutputImage[outindex].y;
	  s_outputVal[index.z].z += s_remainingOpacity[index.z] * renInfo.OutputImage[outindex].z;
	  
	}
	  pos = s_minmaxTrace[index.z].y - s_minmaxTrace[index.z].x;
      }
    }
    pos += volInfo.SampleDistance;
  }

  //write to output

  if(index.x < s_dsize.x && index.y < s_dsize.y){
    renInfo.OutputImage[outindex]=make_uchar4(s_outputVal[index.z].x * 255.0, 
					                          s_outputVal[index.z].y * 255.0, 
					                          s_outputVal[index.z].z * 255.0, 
					                         (1 - s_remainingOpacity[index.z])*255.0);
    renInfo.ZBuffer[outindex]=s_zBuffer[index.z];
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
