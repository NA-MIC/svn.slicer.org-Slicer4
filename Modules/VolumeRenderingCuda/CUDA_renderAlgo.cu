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

__constant__ int c_renderAlgo_size[3];
__constant__ unsigned int c_renderAlgo_dsize[2];
__constant__ float c_renderAlgo_minmax[6];
__constant__ float c_renderAlgo_lightVec[3];

__constant__ float c_renderAlgo_rotationMatrix1[4];
__constant__ float c_renderAlgo_rotationMatrix2[4];
__constant__ float c_renderAlgo_rotationMatrix3[4];
__constant__ float c_renderAlgo_vsize[3];
__constant__ float c_renderAlgo_disp[3];

// Do hybrid rendering (combination between iso-surface rendering and MIP rendering). Here, rendering parameter was set to (1.0-transparencyLevel) MIP and (transparencyLevel) iso-surface volume rendering. Transparency level 1 means fully oblique, and transparency level 0 means fully transparent.

template <typename T>
__global__ void CUDAkernel_renderAlgo_doHybridRender(T* d_sourceData, 
						     float* colorTransferFunction,
						     float* alphaTransferFunction,
						     float* zBuffer,
                                                     T minThreshold, T maxThreshold,
                                                     int sliceDistance, 
                                                     float transparencyLevel, 
                                                     uchar4* d_resultImage,
						     float posX, float posY, float posZ,
						     float focX, float focY, float focZ,
						     float viewX, float viewY, float viewZ)
{
  int xIndex = blockDim.x *blockIdx.x + threadIdx.x;
  int yIndex = blockDim.y *blockIdx.y + threadIdx.y;

  __shared__ float2 s_minmaxTrace[BLOCK_DIM2D*BLOCK_DIM2D]; //starting and ending step of ray tracing 
  __shared__ uchar4 s_resultImage[BLOCK_DIM2D*BLOCK_DIM2D]; //variables to store output image in shared memory 
  __shared__ float s_lensMap[BLOCK_DIM2D*BLOCK_DIM2D*4]; //lens map: position and orientation of ray
  __shared__ float s_rayMap[BLOCK_DIM2D*BLOCK_DIM2D*6]; //ray map: position and orientation of ray after translation and rotation transformation
  __shared__ float s_dsize[3]; //display size (x, y, dummy)
  __shared__ float s_vsize[3]; //voxel dimension
  __shared__ float s_size[3]; //3D data size
  __shared__ float s_disp[3]; //displacement of 3D data in (x,y,z) direction
  __shared__ float s_lightVec[3]; //direction of light
  __shared__ float s_minmax[6]; //region of interest of 3D data (minX, maxX, minY, maxY, minZ, maxZ)
  __shared__ float s_maxVal[BLOCK_DIM2D*BLOCK_DIM2D]; //temporary maximum value (used in MIP) 
  __shared__ int s_pos[BLOCK_DIM2D*BLOCK_DIM2D]; //position of surface (number of steps from camera)
  __shared__ float s_zBuffer[BLOCK_DIM2D*BLOCK_DIM2D]; // z buffer
  float4 s_shadeField; //normal vector of surface

  float test; //temporary variable
  float tempf;
    
  int tempacc=threadIdx.x+threadIdx.y*BLOCK_DIM2D; //index in grid

  __syncthreads();

  //initialization of variables in shared memory

  if(tempacc <3){ 
    s_dsize[xIndex%2]=c_renderAlgo_dsize[xIndex%2];
    s_vsize[xIndex%3]=c_renderAlgo_vsize[xIndex%3];
    s_size[xIndex%3]=c_renderAlgo_size[xIndex%3];
    s_disp[xIndex%3]=c_renderAlgo_disp[xIndex%3];
    s_lightVec[xIndex%3]=c_renderAlgo_lightVec[xIndex%3];
  }else if(tempacc < 9){ 
    s_minmax[xIndex%6]=c_renderAlgo_minmax[xIndex%6];
  }

  __syncthreads();

  int outindex=xIndex+yIndex*s_dsize[0];

  s_maxVal[tempacc]=0;
  s_pos[tempacc]=-1;
  s_resultImage[tempacc]=make_uchar4(0,0,0,0);
  s_zBuffer[tempacc]=zBuffer[outindex];
  
  __syncthreads();

  /*
    camera model start here
  */
  
  s_rayMap[tempacc*6]=posX + s_size[0]*s_vsize[0]/2.0f;
  s_rayMap[tempacc*6+1]=posY + s_size[1]*s_vsize[1]/2.0f;
  s_rayMap[tempacc*6+2]=posZ + s_size[2]*s_vsize[2]/2.0f;
  
  float vecX, vecY, vecZ;
  
  vecX=focX-posX;
  vecY=focY-posY;
  vecZ=focZ-posZ;

  float temp= 1.0f/sqrt(vecX*vecX+vecY*vecY+vecZ*vecZ);
  vecX*=temp;
  vecY*=temp;
  vecZ*=temp;
  
  float verX, verY, verZ;
  float horX, horY, horZ;
  
  float dot = viewX*vecX+viewY*vecY+viewZ*vecZ;

  verX=viewX-dot*vecX;
  verY=viewY-dot*vecY;
  verZ=viewZ-dot*vecZ;

  temp= 1.0f/sqrt(verX*verX+verY*verY+verZ*verZ);
  verX*=temp;
  verY*=temp;
  verZ*=temp;

  horX=verY*vecZ-verZ*vecY;
  horY=verZ*vecX-verX*vecZ;
  horZ=verX*vecY-verY*vecX;

  float posHor=(xIndex-s_dsize[0]*0.5)/s_dsize[0]*0.27;
  float posVer=(yIndex-s_dsize[1]*0.5)/s_dsize[0]*0.27;
  
  s_rayMap[tempacc*6+3]=vecX+posHor*horX+posVer*verX;
  s_rayMap[tempacc*6+4]=vecY+posHor*horY+posVer*verY;
  s_rayMap[tempacc*6+5]=vecZ+posHor*horZ+posVer*verZ;

  /*
    camera model end here
  */
  
  //initialize variables for calculating starting and ending point of ray tracing

  s_minmaxTrace[tempacc].x=-100000.0f;
  s_minmaxTrace[tempacc].y=100000.0f;

  __syncthreads();

  //normalize ray vector

  temp= 1.0f/sqrt((s_rayMap[tempacc*6+3]*s_rayMap[tempacc*6+3]+s_rayMap[tempacc*6+4]*s_rayMap[tempacc*6+4]+s_rayMap[tempacc*6+5]*s_rayMap[tempacc*6+5]));
  s_rayMap[tempacc*6+3]*=temp;
  s_rayMap[tempacc*6+4]*=temp;
  s_rayMap[tempacc*6+5]*=temp;

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

  float val=0;
  float tempData=0;
  float tempx,tempy,tempz;
  int pos=0;

  float initialZBuffer=s_zBuffer[tempacc];

  //perform ray tracing to find maximum intensity and position of surface (steps from camera)

  while((s_minmaxTrace[tempacc].y-s_minmaxTrace[tempacc].x)>=pos){
    
    //calculate current position in ray tracing

    tempx = ( s_rayMap[tempacc*6+0]+((int)s_minmaxTrace[tempacc].x+pos)*s_rayMap[tempacc*6+3]);
    tempy = ( s_rayMap[tempacc*6+1]+((int)s_minmaxTrace[tempacc].x+pos)*s_rayMap[tempacc*6+4]);
    tempz = ( s_rayMap[tempacc*6+2]+((int)s_minmaxTrace[tempacc].x+pos)*s_rayMap[tempacc*6+5]);
    
    tempx /= s_vsize[0];
    tempy /= s_vsize[1];
    tempz /= s_vsize[2];

    if(tempx >= s_minmax[0] && tempx <= s_minmax[1] && tempy >= s_minmax[2] && tempy <= s_minmax[3] && tempz >= s_minmax[4] && tempz <= s_minmax[5] && pos+s_minmaxTrace[tempacc].x >=sliceDistance){ //check whether current position is in front of z buffer wall

      if(pos+s_minmaxTrace[tempacc].x < initialZBuffer){

	temp=d_sourceData[(int)(__float2int_rn(tempz)*s_size[0]*s_size[1]+__float2int_rn(tempy)*s_size[0]+__float2int_rn(tempx))];
	//temp=d_sourceData[(int)((int)tempz*s_size[0]*s_size[1]+(int)tempy*s_size[0]+(int)tempx)];
	
	if(temp>s_maxVal[tempacc])s_maxVal[tempacc]=temp;
	
	if(s_pos[tempacc]==-1 && temp >=minThreshold && temp <= maxThreshold  && pos+s_minmaxTrace[tempacc].x >=sliceDistance){
	  
	  //pos is the position of surface

	  s_pos[tempacc]=pos; 
	  s_zBuffer[tempacc]=pos;  // assign pos to z buffer
	  
	}
      }else{
	
      }
                  
    }
    pos++;
    
  }
  
  // calculating light reflection at object surface

  pos=s_pos[tempacc];
  
  //set current position to surface position

  tempx = ( s_rayMap[tempacc*6+0]+((int)s_minmaxTrace[tempacc].x+pos)*s_rayMap[tempacc*6+3]);
  tempy = ( s_rayMap[tempacc*6+1]+((int)s_minmaxTrace[tempacc].x+pos)*s_rayMap[tempacc*6+4]);
  tempz = ( s_rayMap[tempacc*6+2]+((int)s_minmaxTrace[tempacc].x+pos)*s_rayMap[tempacc*6+5]);
  
  tempx /= s_vsize[0];
  tempy /= s_vsize[1];
  tempz /= s_vsize[2];

  if(s_pos[tempacc]!=-1 ){ // surface was found along current ray
    
    if(__float2int_rd(pos+s_minmaxTrace[tempacc].x) == sliceDistance)val=d_sourceData[(int)(__float2int_rn(tempz)*s_size[0]*s_size[1]+__float2int_rn(tempy)*s_size[0]+__float2int_rn(tempx))];
    
    tempData=d_sourceData[(int)(__float2int_rn(tempz)*s_size[0]*s_size[1]+__float2int_rn(tempy)*s_size[0]+__float2int_rn(tempx))];
    //tempData=d_sourceData[(int)(tempz*s_size[0]*s_size[1]+tempy*s_size[0]+tempx)];

    // calculating normal vector of surface

    s_shadeField.x = ((float)d_sourceData[(int)(__float2int_rn(tempz)*s_size[0]*s_size[1]+__float2int_rn(tempy)*s_size[0]+__float2int_rn(tempx+1))]-(float)d_sourceData[(int)(__float2int_rn(tempz)*s_size[0]*s_size[1]+__float2int_rn(tempy)*s_size[0]+__float2int_rn(tempx-1))]);
    s_shadeField.y = ((float)d_sourceData[(int)(__float2int_rn(tempz)*s_size[0]*s_size[1]+__float2int_rn(tempy+1)*s_size[0]+__float2int_rn(tempx))]-(float)d_sourceData[(int)(__float2int_rn(tempz)*s_size[0]*s_size[1]+__float2int_rn(tempy-1)*s_size[0]+__float2int_rn(tempx))]);
    s_shadeField.z = ((float)d_sourceData[(int)(__float2int_rn(tempz+1)*s_size[0]*s_size[1]+__float2int_rn(tempy)*s_size[0]+__float2int_rn(tempx))]-(float)d_sourceData[(int)(__float2int_rn(tempz-1)*s_size[0]*s_size[1]+__float2int_rn(tempy)*s_size[0]+__float2int_rn(tempx))]);
    
    // normalizing normal vector

    tempf = 1.0/sqrt(SQR(s_shadeField.x) + SQR(s_shadeField.y) + SQR(s_shadeField.z));
    s_shadeField.x = tempf * s_shadeField.x;
    s_shadeField.y = tempf * s_shadeField.y;
    s_shadeField.z = tempf * s_shadeField.z;
    
    if(val==0.0){
      
      // calculating dot product of normal vector and light vector
      
      val = (s_shadeField.x*s_lightVec[0]+s_shadeField.y*s_lightVec[1]+s_shadeField.z*s_lightVec[2]);
    }
    
    if(val<0.0)val=0.0;
    
    if(val<=1.0){
      
      val=(transparencyLevel*val+(1.0-transparencyLevel)*s_maxVal[tempacc]/255.0); //combining maximum intensity and light reflection 
      
      //assigning value into result image using color transfer function
      
      s_resultImage[tempacc]=make_uchar4((unsigned char)( colorTransferFunction[(int)(tempData*3)]*val *255) ,
					 (unsigned char)( colorTransferFunction[(int)(tempData*3+1)]*val *255), 
					 (unsigned char)( colorTransferFunction[(int)(tempData*3+2)]*val *255), 
					 (unsigned char) 255);
      
					 
    }else{ // surface was not found along current ray
      s_resultImage[tempacc]=make_uchar4((unsigned char)val, (unsigned char)val, (unsigned char)val, 255 );
    }
    
  }
  
  //write to output

  d_resultImage[outindex]=s_resultImage[tempacc];
  zBuffer[outindex]=s_zBuffer[tempacc];
}

template <typename T>
__global__ void CUDAkernel_renderAlgo_doIntegrationRender(T* d_sourceData, 
							  float* colorTransferFunction,
							  float* alphaTransferFunction,
							  float* zBuffer,
							  T minThreshold, T maxThreshold,
							  int sliceDistance, 
							  float transparencyLevel, 
							  uchar4* d_resultImage,
							  float posX, float posY, float posZ,
							  float focX, float focY, float focZ,
							  float viewX, float viewY, float viewZ)
{
  int xIndex = blockDim.x *blockIdx.x + threadIdx.x;
  int yIndex = blockDim.y *blockIdx.y + threadIdx.y;

  __shared__ float2 s_minmaxTrace[BLOCK_DIM2D*BLOCK_DIM2D]; //starting and ending step of ray tracing 
  __shared__ float s_rayMap[BLOCK_DIM2D*BLOCK_DIM2D*6]; //ray map: position and orientation of ray after translation and rotation transformation
  __shared__ float s_dsize[3]; //display size (x, y, dummy)
  __shared__ float s_vsize[3]; //voxel dimension
  __shared__ float s_size[3]; //3D data size
  __shared__ float s_minmax[6]; //region of interest of 3D data (minX, maxX, minY, maxY, minZ, maxZ)
  __shared__ float s_integrationVal[BLOCK_DIM2D*BLOCK_DIM2D]; //integration value of alpha
  __shared__ unsigned char s_outputVal[BLOCK_DIM2D*BLOCK_DIM2D*3]; //output value
  __shared__ float s_zBuffer[BLOCK_DIM2D*BLOCK_DIM2D]; // z buffer

  float test;
      
  int tempacc=threadIdx.x+threadIdx.y*BLOCK_DIM2D; //index in grid

  __syncthreads();

  //copying variables into shared memory

  if(tempacc <3){ 
    s_dsize[xIndex%2]=c_renderAlgo_dsize[xIndex%2];
    s_vsize[xIndex%3]=c_renderAlgo_vsize[xIndex%3];
    s_size[xIndex%3]=c_renderAlgo_size[xIndex%3];
  }else if(tempacc < 9){ 
    s_minmax[xIndex%6]=c_renderAlgo_minmax[xIndex%6];
  }

  __syncthreads();

  int outindex=xIndex+yIndex*s_dsize[0]; // index of result image

  //initialization of variables in shared memory

  s_integrationVal[tempacc]=0;
  s_outputVal[tempacc*3]=0;
  s_outputVal[tempacc*3+1]=0;
  s_outputVal[tempacc*3+2]=0;
  s_zBuffer[tempacc]=zBuffer[outindex];
    
  __syncthreads();

  // lens map for perspective projection

  /*
    camera model start here
  */
  
  s_rayMap[tempacc*6]=posX + s_size[0]*s_vsize[0]/2.0f;
  s_rayMap[tempacc*6+1]=posY + s_size[1]*s_vsize[1]/2.0f;
  s_rayMap[tempacc*6+2]=posZ + s_size[2]*s_vsize[2]/2.0f;
  
  float vecX, vecY, vecZ;

  vecX=(focX-posX);
  vecY=(focY-posY);
  vecZ=(focZ-posZ);

  float temp= 1.0f/sqrt(vecX*vecX+vecY*vecY+vecZ*vecZ);
  vecX*=temp;
  vecY*=temp;
  vecZ*=temp;

  float verX, verY, verZ;
  float horX, horY, horZ;
  
  float dot = viewX*vecX+viewY*vecY+viewZ*vecZ;

  verX=viewX-dot*vecX;
  verY=viewY-dot*vecY;
  verZ=viewZ-dot*vecZ;

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
  
  temp= 1.0f/sqrt((s_rayMap[tempacc*6+3]*s_rayMap[tempacc*6+3]+s_rayMap[tempacc*6+4]*s_rayMap[tempacc*6+4]+s_rayMap[tempacc*6+5]*s_rayMap[tempacc*6+5]));
  s_rayMap[tempacc*6+3]*=temp;
  s_rayMap[tempacc*6+4]*=temp;
  s_rayMap[tempacc*6+5]*=temp;

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
    

    if(tempx >= s_minmax[0] && tempx <= s_minmax[1] && tempy >= s_minmax[2] && tempy <= s_minmax[3] && tempz >= s_minmax[4] && tempz <= s_minmax[5] && pos+s_minmaxTrace[tempacc].x >=sliceDistance){ // if current position is in ROI

      if(pos+s_minmaxTrace[tempacc].x < initialZBuffer){ //check whether current position is in front of z buffer wall

	temp=d_sourceData[(int)(__float2int_rn(tempz)*s_size[0]*s_size[1]+__float2int_rn(tempy)*s_size[0]+__float2int_rn(tempx))];

	if( temp >=minThreshold && temp <= maxThreshold){ 

	  alpha=alphaTransferFunction[(int)temp];
	  
	  if(s_zBuffer[tempacc] > pos+s_minmaxTrace[tempacc].x){
	    s_zBuffer[tempacc]=pos+s_minmaxTrace[tempacc].x;
	  }
	  
	  if(s_integrationVal[tempacc]<1.0){ // check if integration value has reached threshold(1.0)
	    if(s_integrationVal[tempacc]+alpha>=1.0)alpha=1.0-s_integrationVal[tempacc]; //make sure that total alpha value does not exceed threshold
	    s_integrationVal[tempacc]+=alpha;
	    s_outputVal[tempacc*3]+=alpha*colorTransferFunction[(int)temp*3]*256.0;
	    s_outputVal[tempacc*3+1]+=alpha*colorTransferFunction[(int)temp*3+1]*256.0;
	    s_outputVal[tempacc*3+2]+=alpha*colorTransferFunction[(int)temp*3+2]*256.0;
	    
	  }else{
	    pos = s_minmaxTrace[tempacc].y-s_minmaxTrace[tempacc].x;
	  }
	}
	

      }else{ // current position is behind z buffer wall
	
	s_outputVal[tempacc*3]+=(1.0-s_integrationVal[tempacc])*d_resultImage[outindex].x;
	s_outputVal[tempacc*3+1]+=(1.0-s_integrationVal[tempacc])*d_resultImage[outindex].y;
	s_outputVal[tempacc*3+2]+=(1.0-s_integrationVal[tempacc])*d_resultImage[outindex].z;
	
	pos = s_minmaxTrace[tempacc].y-s_minmaxTrace[tempacc].x;
	
      }
                  
    }
    pos++;
    
  }

  //write to output

  d_resultImage[outindex]=make_uchar4(s_outputVal[tempacc*3], 
				      s_outputVal[tempacc*3+1], 
				      s_outputVal[tempacc*3+2], 
				      255);
  zBuffer[outindex]=s_zBuffer[tempacc];
}

extern "C"
void CUDArenderAlgo_doRender(uchar4* outputData, //output image
							 cudaRendererInformation* rendererInfo,
							 cudaVolumeInformation* volumeInfo)
{
  // setup execution parameters

  dim3 grid(rendererInfo->Resolution[0] / BLOCK_DIM2D, rendererInfo->Resolution[1]/ BLOCK_DIM2D, 1);
  dim3 threads(BLOCK_DIM2D, BLOCK_DIM2D, 1);

  CUT_DEVICE_INIT();

  // copy host memory to device
  CUDA_SAFE_CALL( cudaMemcpyToSymbol(c_renderAlgo_size, volumeInfo->VolumeSize, sizeof(float)*3, 0));
  CUDA_SAFE_CALL( cudaMemcpyToSymbol(c_renderAlgo_dsize, rendererInfo->Resolution, sizeof(float)*2, 0));
  
  CUDA_SAFE_CALL( cudaMemcpyToSymbol(c_renderAlgo_vsize, volumeInfo->VoxelSize, sizeof(float)*3, 0));
  CUDA_SAFE_CALL( cudaMemcpyToSymbol(c_renderAlgo_disp, volumeInfo->VolumeTransformation, sizeof(float)*3, 0));

  CUDA_SAFE_CALL( cudaMemcpyToSymbol(c_renderAlgo_minmax, volumeInfo->MinMaxValue, sizeof(float)*6, 0));
  CUDA_SAFE_CALL( cudaMemcpyToSymbol(c_renderAlgo_lightVec, rendererInfo->LightVectors, sizeof(float)*3 * rendererInfo->LightCount, 0));

  // execute the kernel
  // Switch to various rendering methods.
  float transparencyLevel = 1.0;
  
  CUDAkernel_renderAlgo_doIntegrationRender<<< grid, threads >>>( \
	 (unsigned char*)volumeInfo->SourceData, \
	 volumeInfo->ColorTransferFunction, \
	 volumeInfo->AlphaTransferFunction, \
	 rendererInfo->ZBuffer, \
	 (unsigned char)volumeInfo->MinThreshold, (unsigned char)volumeInfo->MaxThreshold,	\
	 rendererInfo->NearPlane, \
	 transparencyLevel, \
	 outputData,\
	 rendererInfo->CameraPos[0], rendererInfo->CameraPos[1], rendererInfo->CameraPos[2], \
	 rendererInfo->TargetPos[0], rendererInfo->TargetPos[1], rendererInfo->TargetPos[2], \
     rendererInfo->ViewUp[0], rendererInfo->ViewUp[1], rendererInfo->ViewUp[2]);
  
  /*
#define CUDA_KERNEL_CALL(ID, TYPE)   \
	if (inputDataType == ID) \
	 CUDAkernel_renderAlgo_doIntegrationRender<<< grid, threads >>>( \
	 (TYPE*)renderData, \
	 colorTransferFunction, \
	 alphaTransferFunction, \
	 zBuffer, \
	 minThreshold, maxThreshold, \
	 sliceDistance, \
	 transparencyLevel, \
	 outputData)

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
