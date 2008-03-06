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
                         T val1, T val2, T val3, T val4,
                         T val5, T val6, T val7, T val8)
{
    float revX = 1-posX;
    float revY = 1-posY;
    float revZ = 1-posZ;

    return ((T) 
        (revX * (revY * (revZ * val1  +
        posZ * val2) +
        posY * (revZ * val3  +
        posZ * val4))+
        posX * (revY * (revZ * val5  +
        posZ * val6)   +
        posY * (revZ * val7 +
        posZ * val8)))
        );
}

__device__ void MatMul(const float mat[4][4], float3* out, float inX, float inY, float inZ)
{
    out->x = mat[0][0] * inX + mat[0][1] * inY + mat[0][2] * inZ + mat[0][3] * 1.0;
    out->y = mat[1][0] * inX + mat[1][1] * inY + mat[1][2] * inZ + mat[1][3] * 1.0;
    out->z = mat[2][0] * inX + mat[2][1] * inY + mat[2][2] * inZ + mat[2][3] * 1.0;
}

__device__ void CUDAkernel_SetRayMap(const int3& index, float* raymap, const cudaRendererInformation& renInfo)
{
    float rayLength;
    float posHor= (float)index.x / (float)renInfo.Resolution.x;
    float posVer= (float)index.y / (float)renInfo.Resolution.y;

    raymap[index.z*6]   = renInfo.CameraRayStart.x  + renInfo.CameraRayStartX.x * posVer + renInfo.CameraRayStartY.x * posHor;
    raymap[index.z*6+1] = renInfo.CameraRayStart.y  + renInfo.CameraRayStartX.y * posVer + renInfo.CameraRayStartY.y * posHor;
    raymap[index.z*6+2] = renInfo.CameraRayStart.z  + renInfo.CameraRayStartX.z * posVer + renInfo.CameraRayStartY.z * posHor;

    // Ray Length
    raymap[index.z*6+3] = (renInfo.CameraRayEnd.x  + renInfo.CameraRayEndX.x * posVer + renInfo.CameraRayEndY.x * posHor) - raymap[index.z*6];
    raymap[index.z*6+4] = (renInfo.CameraRayEnd.y  + renInfo.CameraRayEndX.y * posVer + renInfo.CameraRayEndY.y * posHor) - raymap[index.z*6+1];
    raymap[index.z*6+5] = (renInfo.CameraRayEnd.z  + renInfo.CameraRayEndX.z * posVer + renInfo.CameraRayEndY.z * posHor) - raymap[index.z*6+2];

    rayLength = sqrtf(raymap[index.z*6+3] * raymap[index.z*6+3] + 
                      raymap[index.z*6+4] * raymap[index.z*6+4] + 
                      raymap[index.z*6+5] * raymap[index.z*6+5]);

    // Normalize the direction vector
    raymap[index.z*6+3] /= rayLength;
    raymap[index.z*6+4] /= rayLength;
    raymap[index.z*6+5] /= rayLength;
}

template <typename T>
__device__ void CUDAkernel_RayCastAlgorithm(const int3& index,
                                            int outindex,
                                            const float* minmax /*[6] */,
                                            const float* rayMap,
                                            const cudaVolumeInformation& volInfo,
                                            const cudaRendererInformation& renInfo,
                                            float3* outputVal,
                                            float* zBuffer,
                                            float* remainingOpacity)
{
    //ray tracing start from here
    float depth = 0.0;  //current step distance from camera

    float3 tempPos;     //!< variables to store current position
    float  distFromCam; //!< The distance from the camera to the Image
    T tempValue;        //!< A Temporary color value
    int tempIndex;      //!< Temporaty index in the 3D data
    float alpha;        //!< Alpha value of current voxel
    float initialZBuffer = zBuffer[index.z]; //!< initial zBuffer from input

    float A = renInfo.ClippingRange.y / (renInfo.ClippingRange.y - renInfo.ClippingRange.x);
    float B = renInfo.ClippingRange.y * renInfo.ClippingRange.x / (renInfo.ClippingRange.x - renInfo.ClippingRange.y);


    //perform ray tracing until integration of alpha value reach threshold 
    while(depth < 1.0) {
        distFromCam = B / ( depth - A);

        //calculate current position in ray tracing
        MatMul(volInfo.Transform, &tempPos, 
            (renInfo.CameraPos.x + distFromCam * rayMap[index.z*6+3]),
            (renInfo.CameraPos.y + distFromCam * rayMap[index.z*6+4]),
            (renInfo.CameraPos.z + distFromCam * rayMap[index.z*6+5]));

        // if current position is in ROI
        if(tempPos.x >= minmax[0] && tempPos.x < minmax[1] &&
           tempPos.y >= minmax[2] && tempPos.y < minmax[3] &&
           tempPos.z >= minmax[4] && tempPos.z < minmax[5] )
        {
            //check whether current position is in front of z buffer wall
            if(depth < 1.0 )//initialZBuffer)
            { 

                //tempValue=((T*)volInfo.SourceData)[(int)(__float2int_rn(tempPos.z)*volInfo.VolumeSize.x*volInfo.VolumeSize.y + 
                //    __float2int_rn(tempPos.y)*volInfo.VolumeSize.x +
                //    __float2int_rn(tempPos.x))];
                /*interpolation start here*/
                float posX = tempPos.x - __float2int_rd(tempPos.x);
                float posY = tempPos.y - __float2int_rd(tempPos.y);
                float posZ = tempPos.z - __float2int_rd(tempPos.z);


               /* tempValue=interpolate((float)0,(float)0,(float)0,
                    ((T*)volInfo.SourceData)[(int)((int)(tempPos.z)*volInfo.VolumeSize.x*volInfo.VolumeSize.y + 
                    (int)(tempPos.y)*volInfo.VolumeSize.x + 
                    (int)(tempPos.x))],
                    (T)0,(T)0,(T)0,(T)0,(T)0,(T)0,(T)0);*/

                int base = __float2int_rd((tempPos.z)) * volInfo.VolumeSize.x*volInfo.VolumeSize.y + 
                           __float2int_rd((tempPos.y)) * volInfo.VolumeSize.x + 
                           __float2int_rd((tempPos.x));

                tempValue=interpolate(posX, posY, posZ,
                ((T*)volInfo.SourceData)[base],
                ((T*)volInfo.SourceData)[(int)(base + volInfo.VolumeSize.x*volInfo.VolumeSize.y)],
                ((T*)volInfo.SourceData)[(int)(base + volInfo.VolumeSize.x)],
                ((T*)volInfo.SourceData)[(int)(base + volInfo.VolumeSize.x*volInfo.VolumeSize.y + volInfo.VolumeSize.x)],
                ((T*)volInfo.SourceData)[(int)(base + 1)],
                ((T*)volInfo.SourceData)[(int)(base + volInfo.VolumeSize.x*volInfo.VolumeSize.y + 1)],
                ((T*)volInfo.SourceData)[(int)(base + volInfo.VolumeSize.x + 1)],
                ((T*)volInfo.SourceData)[(int)(base + volInfo.VolumeSize.x*volInfo.VolumeSize.y + volInfo.VolumeSize.x + 1)]);
                /*interpolation end here*/

                tempIndex = __float2int_rn((volInfo.FunctionSize-1) * (float)(tempValue-volInfo.FunctionRange[0]) /
                                                                      (float)(volInfo.FunctionRange[1]-volInfo.FunctionRange[0]));
                alpha = volInfo.AlphaTransferFunction[tempIndex];
                if(alpha >= 0){

                    if(remainingOpacity[index.z] > 0.02)  // check if remaining opacity has reached threshold(0.02)
                    {
                        outputVal[index.z].x += remainingOpacity[index.z] * alpha * volInfo.ColorTransferFunction[tempIndex*3];
                        outputVal[index.z].y += remainingOpacity[index.z] * alpha * volInfo.ColorTransferFunction[tempIndex*3+1];
                        outputVal[index.z].z += remainingOpacity[index.z] * alpha * volInfo.ColorTransferFunction[tempIndex*3+2];
                        remainingOpacity[index.z] *= (1.0 - alpha);
                    }
                    else // buffer filled to the max value
                    { 
                        zBuffer[index.z] = depth;
                        break;
                    }
                }

            } 
            else 
            { // current position is behind z buffer wall
                if(index.x < renInfo.Resolution.x && index.y < renInfo.Resolution.y)
                {
                    outputVal[index.z].x += remainingOpacity[index.z] * renInfo.OutputImage[outindex].x;
                    outputVal[index.z].y += remainingOpacity[index.z] * renInfo.OutputImage[outindex].y;
                    outputVal[index.z].z += remainingOpacity[index.z] * renInfo.OutputImage[outindex].z;
                }
                break;
            }
        }
        depth += .002 * volInfo.SampleDistance;
    }
}

__device__ CUDAkernel_WriteData(
{
    if(index.x < renInfo.Resolution.x && index.y < renInfo.Resolution.y)
    {
        renInfo.OutputImage[outindex] = make_uchar4(s_outputVal[index.z].x * 255.0, 
                                                    s_outputVal[index.z].y * 255.0, 
                                                    s_outputVal[index.z].z * 255.0, 
                                                    (1 - s_remainingOpacity[index.z]) * 255.0);
        //renInfo.ZBuffer[renInfo.Resolution.x - index.x + index.y * renInfo.Resolution.x] = s_zBuffer[index.z];
    }
}

__constant__ cudaVolumeInformation   volInfo;
__constant__ cudaRendererInformation renInfo;

template <typename T>
__global__ void CUDAkernel_renderAlgo_doIntegrationRender()
{
    __shared__ float           s_rayMap[BLOCK_DIM2D*BLOCK_DIM2D*6];         //ray map: position and orientation of ray after translation and rotation transformation
    __shared__ float           s_minmax[6];                                 //region of interest of 3D data (minX, maxX, minY, maxY, minZ, maxZ)
    __shared__ float3          s_outputVal[BLOCK_DIM2D*BLOCK_DIM2D];        //output value
    __shared__ float           s_remainingOpacity[BLOCK_DIM2D*BLOCK_DIM2D]; //integration value of alpha
    __shared__ float           s_zBuffer[BLOCK_DIM2D*BLOCK_DIM2D];          // z buffer

    int3 index;
    index.x = blockDim.x *blockIdx.x + threadIdx.x;
    index.y = blockDim.y *blockIdx.y + threadIdx.y;
    index.z = threadIdx.x + threadIdx.y * BLOCK_DIM2D; //index in grid

    //copying variables into shared memory
    if(index.z < 3){ 
    }else if(index.z < 9){ 
        s_minmax[index.x%6] = volInfo.MinMaxValue[index.x%6];
    }
    s_outputVal[index.z].x = 0.0f;
    s_outputVal[index.z].y = 0.0f;
    s_outputVal[index.z].z = 0.0f;

    //initialization of variables in shared memory
    int outindex = index.x + index.y * renInfo.Resolution.x; // index of result image
    s_remainingOpacity[index.z] = 1.0;
    if(index.x < renInfo.Resolution.x && index.y < renInfo.Resolution.y){
        s_zBuffer[index.z] = renInfo.ZBuffer[renInfo.Resolution.x - index.x + index.y * renInfo.Resolution.x];// (renInfo.ClippingRange.y * renInfo.ClippingRange.x / (renInfo.ClippingRange.x - renInfo.ClippingRange.y)) / (renInfo.ZBuffer[outindex] - renInfo.ClippingRange.y / (renInfo.ClippingRange.y - renInfo.ClippingRange.x));
    } else /* outside of screen */ {
        s_zBuffer[index.z]=0;
    }

    CUDAkernel_SetRayMap(index, s_rayMap, renInfo);

    __syncthreads();

    // Call the Algorithm
    CUDAkernel_RayCastAlgorithm<T>(index, outindex, s_minmax /*[6] */,
                                   s_rayMap, volInfo, renInfo,
                                   s_outputVal, s_zBuffer, s_remainingOpacity);

    //write to output
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

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(volInfo, &volumeInfo, sizeof(cudaVolumeInformation)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(renInfo, &rendererInfo, sizeof(cudaRendererInformation)));

    CUT_DEVICE_INIT();

    // The CUDA Kernel Function Definition, so we do not have to write it down below
#define CUDA_KERNEL_CALL(ID, TYPE)   \
    if (volumeInfo.InputDataType == ID) \
    CUDAkernel_renderAlgo_doIntegrationRender<TYPE> <<< grid, threads >>>()

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
