// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

extern "C" {
#include "CUDA_renderAlgo.h"
}

// includes, project
#include <cutil.h>
//#include "vtkType.h"
// includes, kernels

#define BLOCK_DIM2D 16// this must be set to 4 or more
#define SQR(X) ((X) * (X) )

__device__ unsigned char interpolate(float posX, float posY, float posZ,
				     unsigned char val1,
				     unsigned char val2,
				     unsigned char val3,
				     unsigned char val4,
				     unsigned char val5,
				     unsigned char val6,
				     unsigned char val7,
				     unsigned char val8){
  float revX=1-posX;
  float revY=1-posY;
  float revZ=1-posZ;
  
  return ((unsigned char) (revX*revY*revZ* val1+
			   revX*revY*posZ* val2+
			   revX*posY*revZ* val3+
			   revX*posY*posZ* val4+
			   posX*revY*revZ* val5+
			   posX*revY*posZ* val6+
			   posX*posY*revZ* val7+
			   posX*posY*posZ* val8)
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

  __shared__ float2 s_minmaxTrace[BLOCK_DIM2D*BLOCK_DIM2D]; //starting and ending step of ray tracing 
  __shared__ float s_rayMap[BLOCK_DIM2D*BLOCK_DIM2D*6]; //ray map: position and orientation of ray after translation and rotation transformation
  __shared__ float s_dsize[3]; //display size (x, y, dummy)
  __shared__ float s_vsize[3]; //voxel dimension
  __shared__ float s_size[3]; //3D data size
  __shared__ float s_minmax[6]; //region of interest of 3D data (minX, maxX, minY, maxY, minZ, maxZ)
  __shared__ float s_remainingOpacity[BLOCK_DIM2D*BLOCK_DIM2D]; //integration value of alpha
  __shared__ unsigned char s_outputVal[BLOCK_DIM2D*BLOCK_DIM2D*3]; //output value
  __shared__ float s_zBuffer[BLOCK_DIM2D*BLOCK_DIM2D]; // z buffer

  float test;

  int tempacc=threadIdx.x+threadIdx.y*BLOCK_DIM2D; //index in grid

  __syncthreads();

  //copying variables into shared memory

  if(tempacc <3){ 
    s_dsize[xIndex%2]=renInfo.Resolution[xIndex%2];
    s_vsize[xIndex%3]=volInfo.VoxelSize[xIndex%3];
    s_size[xIndex%3]=volInfo.VolumeSize[xIndex%3];
  }else if(tempacc < 9){ 
    s_minmax[xIndex%6]=volInfo.MinMaxValue[xIndex%6];
  }

  __syncthreads();

  int outindex=xIndex+yIndex*s_dsize[0]; // index of result image

  //initialization of variables in shared memory

  s_remainingOpacity[tempacc]=1.0;
  s_outputVal[tempacc*3]=0;
  s_outputVal[tempacc*3+1]=0;
  s_outputVal[tempacc*3+2]=0;
  s_zBuffer[tempacc]=renInfo.ZBuffer[outindex]; //renInfo.ClippingRange[0] + renInfo.ZBuffer[outindex] * (renInfo.ClippingRange[1] - renInfo.ClippingRange[0]);
    
  __syncthreads();

  // lens map for perspective projection

  /*
    camera model start here
  */
  
  s_rayMap[tempacc*6]=renInfo.CameraPos[0] ;//+ s_size[0]*s_vsize[0]/2.0f;
  s_rayMap[tempacc*6+1]=renInfo.CameraPos[1] ;//+ s_size[1]*s_vsize[1]/2.0f;
  s_rayMap[tempacc*6+2]=renInfo.CameraPos[2] ;//+ s_size[2]*s_vsize[2]/2.0f;
  
  float vecX, vecY, vecZ;

  vecX=(renInfo.TargetPos[0]-renInfo.CameraPos[0]);
  vecY=(renInfo.TargetPos[1]-renInfo.CameraPos[1]);
  vecZ=(renInfo.TargetPos[2]-renInfo.CameraPos[2]);

  float temp= 1.0f/sqrt(vecX*vecX+vecY*vecY+vecZ*vecZ);
  vecX*=temp;
  vecY*=temp;
  vecZ*=temp;

  float verX, verY, verZ;
  float horX, horY, horZ;
  
  float dot = renInfo.ViewUp[0]*vecX+renInfo.ViewUp[1]*vecY+renInfo.ViewUp[2]*vecZ;

  verX=renInfo.ViewUp[0]-dot*vecX;
  verY=renInfo.ViewUp[1]-dot*vecY;
  verZ=renInfo.ViewUp[2]-dot*vecZ;

  temp= 1.0f/sqrt(verX*verX+verY*verY+verZ*verZ);
  verX*=temp;
  verY*=temp;
  verZ*=temp;

  horX=verY*vecZ-verZ*vecY;
  horY=verZ*vecX-verX*vecZ;
  horZ=verX*vecY-verY*vecX;

  float posHor=(xIndex-s_dsize[0]*0.5)/s_dsize[0]*0.27;
  float posVer=(yIndex-s_dsize[1]*0.5)/s_dsize[0]*0.27;
  
  s_rayMap[tempacc*6+3]=(vecX+posHor*horX+posVer*verX);
  s_rayMap[tempacc*6+4]=(vecY+posHor*horY+posVer*verY);
  s_rayMap[tempacc*6+5]=(vecZ+posHor*horZ+posVer*verZ);

  /*
    camera model end here
  */
 
  //initialize variables for calculating starting and ending point of ray tracing

  s_minmaxTrace[tempacc].x=-100000.0f;
  s_minmaxTrace[tempacc].y=100000.0f;

  __syncthreads();
  
  //normalize ray vector

  float getmax = fabs(s_rayMap[tempacc*6+3]/s_vsize[0]);
  if(fabs(s_rayMap[tempacc*6+4]/s_vsize[1])>getmax) getmax = fabs(s_rayMap[tempacc*6+4]/s_vsize[1]);
  if(fabs(s_rayMap[tempacc*6+5]/s_vsize[2])>getmax) getmax = fabs(s_rayMap[tempacc*6+5]/s_vsize[2]);

  if(getmax!=0){
    float temp= 1.0f/getmax;
    s_rayMap[tempacc*6+3]*=temp;
    s_rayMap[tempacc*6+4]*=temp;
    s_rayMap[tempacc*6+5]*=temp;
  }

  float stepSize=sqrt(s_rayMap[tempacc*6+3]*s_rayMap[tempacc*6+3]+s_rayMap[tempacc*6+4]*s_rayMap[tempacc*6+4]+s_rayMap[tempacc*6+5]*s_rayMap[tempacc*6+5]);
  
  __syncthreads();

  //calculating starting and ending point of ray tracing

 if(s_rayMap[tempacc*6+3] > 1.0e-3){
    s_minmaxTrace[tempacc].y = ( ((s_minmax[1]-2)*s_vsize[0]-s_rayMap[tempacc*6])/s_rayMap[tempacc*6+3] );
    s_minmaxTrace[tempacc].x = ( ((s_minmax[0]+2)*s_vsize[0]-s_rayMap[tempacc*6])/s_rayMap[tempacc*6+3] );
  }
  else if(s_rayMap[tempacc*6+3] < -1.0e-3){
    s_minmaxTrace[tempacc].x = ( ((s_minmax[1]-2)*s_vsize[0]-s_rayMap[tempacc*6])/s_rayMap[tempacc*6+3] );
    s_minmaxTrace[tempacc].y = ( ((s_minmax[0]+2)*s_vsize[0]-s_rayMap[tempacc*6])/s_rayMap[tempacc*6+3] );
  }
  
  if(s_rayMap[tempacc*6+4] > 1.0e-3){
    test = ( ((s_minmax[3]-2)*s_vsize[1]-s_rayMap[tempacc*6+1])/s_rayMap[tempacc*6+4] );
    if( test < s_minmaxTrace[tempacc].y){
      s_minmaxTrace[tempacc].y = test;
    }
    test = ( ((s_minmax[2]+2)*s_vsize[1]-s_rayMap[tempacc*6+1])/s_rayMap[tempacc*6+4] );
    if( test > s_minmaxTrace[tempacc].x){
      s_minmaxTrace[tempacc].x = test;
    }
  }
  else if(s_rayMap[tempacc*6+4] < -1.0e-3){
    test = ( ((s_minmax[3]-2)*s_vsize[1]-s_rayMap[tempacc*6+1])/s_rayMap[tempacc*6+4] );
    if( test > s_minmaxTrace[tempacc].x){
      s_minmaxTrace[tempacc].x = test;
    }
    test = ( ((s_minmax[2]+2)*s_vsize[1]-s_rayMap[tempacc*6+1])/s_rayMap[tempacc*6+4] );
    if( test < s_minmaxTrace[tempacc].y){
      s_minmaxTrace[tempacc].y = test;
    }
  }
  

  if(s_rayMap[tempacc*6+5] > 1.0e-3){
    test = ( ((s_minmax[5]-2)*s_vsize[2]-s_rayMap[tempacc*6+2])/s_rayMap[tempacc*6+5] );
    if( test < s_minmaxTrace[tempacc].y){
      s_minmaxTrace[tempacc].y = test;
    }
    test = ( ((s_minmax[4]+2)*s_vsize[2]-s_rayMap[tempacc*6+2])/s_rayMap[tempacc*6+5] );
    if( test > s_minmaxTrace[tempacc].x){
      s_minmaxTrace[tempacc].x = test;
    }
  }
  else if(s_rayMap[tempacc*6+5] < -1.0e-3){
    test = ( ((s_minmax[5]-2)*s_vsize[2]-s_rayMap[tempacc*6+2])/s_rayMap[tempacc*6+5] );
    if( test > s_minmaxTrace[tempacc].x){
      s_minmaxTrace[tempacc].x = test;
    }
    test = ( ((s_minmax[4]+2)*s_vsize[2]-s_rayMap[tempacc*6+2])/s_rayMap[tempacc*6+5] );
    if( test < s_minmaxTrace[tempacc].y){
      s_minmaxTrace[tempacc].y = test;
    }
  }
  __syncthreads();

  //ray tracing start from here

  float tempx,tempy,tempz; // variables to store current position
  int pos=0; //current step distance from camera

  //float temp; //temporary variable to store data during calculation
  float alpha; //alpha value of current voxel
  float initialZBuffer=s_zBuffer[tempacc]; //initial zBuffer from input

  //perform ray tracing until integration of alpha value reach threshold 
  
  while((s_minmaxTrace[tempacc].y-s_minmaxTrace[tempacc].x)>=pos){
    
    //calculate current position in ray tracing

    tempx = ( s_rayMap[tempacc*6+0]+((int)s_minmaxTrace[tempacc].x+pos)*s_rayMap[tempacc*6+3]);
    tempy = ( s_rayMap[tempacc*6+1]+((int)s_minmaxTrace[tempacc].x+pos)*s_rayMap[tempacc*6+4]);
    tempz = ( s_rayMap[tempacc*6+2]+((int)s_minmaxTrace[tempacc].x+pos)*s_rayMap[tempacc*6+5]);
    
    tempx /= s_vsize[0];
    tempy /= s_vsize[1];
    tempz /= s_vsize[2];
    

    if(tempx >= s_minmax[0] && tempx <= s_minmax[1] && tempy >= s_minmax[2] && tempy <= s_minmax[3] && tempz >= s_minmax[4] && tempz <= s_minmax[5] && pos+s_minmaxTrace[tempacc].x >= -500 /*renInfo.ClippingRange[0]*/){ // if current position is in ROI

      if((pos+s_minmaxTrace[tempacc].x)*stepSize < initialZBuffer){ //check whether current position is in front of z buffer wall

	temp=((T*)volInfo.SourceData)[(int)(__float2int_rn(tempz)*s_size[0]*s_size[1]+__float2int_rn(tempy)*s_size[0]+__float2int_rn(tempx))];
	
	/*interpolation start here*/
	/*
	float posX=tempx-(int)tempx;
	float posY=tempy-(int)tempy;
	float posZ=tempz-(int)tempz;

	temp=interpolate(posX, posY, posZ,
			 ((T*)volInfo.SourceData)[(int)((int)(tempz)*s_size[0]*s_size[1]+(int)(tempy)*s_size[0]+(int)(tempx))],
			 ((T*)volInfo.SourceData)[(int)((int)(tempz+1)*s_size[0]*s_size[1]+(int)(tempy)*s_size[0]+(int)(tempx))],
			 ((T*)volInfo.SourceData)[(int)((int)(tempz)*s_size[0]*s_size[1]+(int)(tempy+1)*s_size[0]+(int)(tempx))],
			 ((T*)volInfo.SourceData)[(int)((int)(tempz+1)*s_size[0]*s_size[1]+(int)(tempy+1)*s_size[0]+(int)(tempx))],
			 ((T*)volInfo.SourceData)[(int)((int)(tempz)*s_size[0]*s_size[1]+(int)(tempy)*s_size[0]+(int)(tempx+1))],
			 ((T*)volInfo.SourceData)[(int)((int)(tempz+1)*s_size[0]*s_size[1]+(int)(tempy)*s_size[0]+(int)(tempx+1))],
			 ((T*)volInfo.SourceData)[(int)((int)(tempz)*s_size[0]*s_size[1]+(int)(tempy+1)*s_size[0]+(int)(tempx+1))],
			 ((T*)volInfo.SourceData)[(int)((int)(tempz+1)*s_size[0]*s_size[1]+(int)(tempy+1)*s_size[0]+(int)(tempx+1))]);
	*/

	/*interpolation end here*/

	if( temp >=(T)volInfo.MinThreshold && temp <= (T)volInfo.MaxThreshold){ 

	  alpha=volInfo.AlphaTransferFunction[(int)temp];
	  
	  if(s_zBuffer[tempacc] > (pos+s_minmaxTrace[tempacc].x)*stepSize){
	    s_zBuffer[tempacc]=(pos+s_minmaxTrace[tempacc].x)*stepSize;
	  }
	  if(s_remainingOpacity[tempacc]>0.02){ // check if remaining opacity has reached threshold(0.02)
	    s_outputVal[tempacc*3]+=s_remainingOpacity[tempacc]*alpha*volInfo.ColorTransferFunction[(int)temp*3]*256.0;
	    s_outputVal[tempacc*3+1]+=s_remainingOpacity[tempacc]*alpha*volInfo.ColorTransferFunction[(int)temp*3+1]*256.0;
	    s_outputVal[tempacc*3+2]+=s_remainingOpacity[tempacc]*alpha*volInfo.ColorTransferFunction[(int)temp*3+2]*256.0;
	    s_remainingOpacity[tempacc]*=(1.0-alpha);
	  }else{
	    pos = s_minmaxTrace[tempacc].y-s_minmaxTrace[tempacc].x;
	  }
	}
	

      }else{ // current position is behind z buffer wall
	
	s_outputVal[tempacc*3]+=(s_remainingOpacity[tempacc])*renInfo.OutputImage[outindex].x;
	s_outputVal[tempacc*3+1]+=(s_remainingOpacity[tempacc])*renInfo.OutputImage[outindex].y;
	s_outputVal[tempacc*3+2]+=(s_remainingOpacity[tempacc])*renInfo.OutputImage[outindex].z;
	
	pos = s_minmaxTrace[tempacc].y-s_minmaxTrace[tempacc].x;
	
      }
                  
    }
    pos++;
    
  }

  //write to output

  renInfo.OutputImage[outindex]=make_uchar4(s_outputVal[tempacc*3], 
                                            s_outputVal[tempacc*3+1], 
					    s_outputVal[tempacc*3+2], 
					    (1-s_remainingOpacity[tempacc])*255.0);
  renInfo.ZBuffer[outindex]=s_zBuffer[tempacc];
}

extern "C"
void CUDArenderAlgo_doRender(const cudaRendererInformation& rendererInfo,
							 const cudaVolumeInformation& volumeInfo)
{
  // setup execution parameters

  dim3 grid(rendererInfo.Resolution[0] / BLOCK_DIM2D, rendererInfo.Resolution[1]/ BLOCK_DIM2D, 1);
  dim3 threads(BLOCK_DIM2D, BLOCK_DIM2D, 1);

  CUT_DEVICE_INIT();

  // execute the kernel
  // Switch to various rendering methods.
  //float transparencyLevel = 1.0;
  
  CUDAkernel_renderAlgo_doIntegrationRender<unsigned char> <<< grid, threads >>>( \
	 rendererInfo,
	 volumeInfo)  
  /*
#define CUDA_KERNEL_CALL(ID, TYPE)   \
	if (inputDataType == ID) \
	 CUDAkernel_renderAlgo_doIntegrationRender<<< grid, threads >>>( \
	 outputData, \
	 colorTransferFunction, \
	 alphaTransferFunction, \
	 zBuffer, \
	 minThreshold, maxThreshold, \
	 sliceDistance, \
	 transparencyLevel)

// Add all the other types.
  CUDA_KERNEL_CALL(VTK_UNSIGNED_CHAR, unsigned char);
  else CUDA_KERNEL_CALL(VTK_CHAR, char);
  else CUDA_KERNEL_CALL(VTK_SHORT, short);
  else CUDA_KERNEL_CALL(VTK_UNSIGNED_SHORT, unsigned short);
  else CUDA_KERNEL_CALL(VTK_FLOAT, float);
  else CUDA_KERNEL_CALL(VTK_DOUBLE, double);
  else CUDA_KERNEL_CALL(VTK_INT, int);
  */


  CUT_CHECK_ERROR("Kernel execution failed");

  return;
}
