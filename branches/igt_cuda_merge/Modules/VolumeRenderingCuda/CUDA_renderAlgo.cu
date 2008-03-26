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

#include "CUDA_renderRayCastComposite.h"
#include "CUDA_renderRayCastMIP.h"
#include "CUDA_renderRayCastIsosurface.h"

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

__device__ void MatMul(const float mat[4][4], float3* out, float inX, float inY, float inZ)
{
    out->x = mat[0][0] * inX + mat[0][1] * inY + mat[0][2] * inZ + mat[0][3] * 1.0;
    out->y = mat[1][0] * inX + mat[1][1] * inY + mat[1][2] * inZ + mat[1][3] * 1.0;
    out->z = mat[2][0] * inX + mat[2][1] * inY + mat[2][2] * inZ + mat[2][3] * 1.0;
}

__device__ void CUDAkernel_SetRayMap(const int3& index, float3* raymap, const cudaRendererInformation& renInfo)
{
    float rayLength;
    float posHor= (float)index.x / (float)renInfo.Resolution.x;
    float posVer= (float)index.y / (float)renInfo.Resolution.y;

    raymap[index.z*2].x = renInfo.CameraRayStart.x  + renInfo.CameraRayStartX.x * posVer + renInfo.CameraRayStartY.x * posHor;
    raymap[index.z*2].y = renInfo.CameraRayStart.y  + renInfo.CameraRayStartX.y * posVer + renInfo.CameraRayStartY.y * posHor;
    raymap[index.z*2].z = renInfo.CameraRayStart.z  + renInfo.CameraRayStartX.z * posVer + renInfo.CameraRayStartY.z * posHor;

    // Ray Length
    raymap[index.z*2+1].x = (renInfo.CameraRayEnd.x  + renInfo.CameraRayEndX.x * posVer + renInfo.CameraRayEndY.x * posHor) - raymap[index.z*2].x;
    raymap[index.z*2+1].y = (renInfo.CameraRayEnd.y  + renInfo.CameraRayEndX.y * posVer + renInfo.CameraRayEndY.y * posHor) - raymap[index.z*2].y;
    raymap[index.z*2+1].z = (renInfo.CameraRayEnd.z  + renInfo.CameraRayEndX.z * posVer + renInfo.CameraRayEndY.z * posHor) - raymap[index.z*2].z;

    rayLength = sqrtf(raymap[index.z*2+1].x * raymap[index.z*2+1].x + 
                      raymap[index.z*2+1].y * raymap[index.z*2+1].y + 
                      raymap[index.z*2+1].z * raymap[index.z*2+1].z);

    // Normalize the direction vector
    raymap[index.z*2+1].x /= rayLength;
    raymap[index.z*2+1].y /= rayLength;
    raymap[index.z*2+1].z /= rayLength;
}

__device__ void CUDAkernel_WriteData(const int3& index, int outindex,
                                const float3* outputVal, const float* remainingOpacity,
                                const float* zBuffer,
                                cudaRendererInformation& renInfo)
{
    if(index.x < renInfo.Resolution.x && index.y < renInfo.Resolution.y)
    {
        renInfo.OutputImage[outindex] = make_uchar4(outputVal[index.z].x * 255.0, 
                                                    outputVal[index.z].y * 255.0, 
                                                    outputVal[index.z].z * 255.0, 
                                                    (1 - remainingOpacity[index.z]) * 255.0);
        //renInfo.ZBuffer[renInfo.Resolution.x - index.x + index.y * renInfo.Resolution.x] = zBuffer[index.z];
    }
}

__constant__ cudaVolumeInformation   volInfo;
__constant__ cudaRendererInformation renInfo;

template <typename T, class ALGORITHM>
__global__ void CUDAkernel_renderAlgo_doIntegrationRender()
{
    __shared__ float3          s_rayMap[BLOCK_DIM2D*BLOCK_DIM2D*2];         //ray map: position and orientation of ray after translation and rotation transformation
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

    // Call the Algorithm (Composite or MIP or Isosurface)
    ALGORITHM algo;
    algo(index, outindex, s_minmax /*[6] */,
                                   s_rayMap, volInfo, renInfo,
                                   s_outputVal, s_zBuffer, s_remainingOpacity);

    //write to output
    CUDAkernel_WriteData(index, outindex, 
                        s_outputVal, s_remainingOpacity,
                        s_zBuffer, renInfo);
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

    //CUDAkernel_RayCastMIPAlgorithm
    //CUDAkernel_RayCastCompositeAlgorithm
    //CUDAkernel_RayCastIsosurfaceAlgorithm
    // CUDAkernel_Interpolate_Trilinear
    // CUDAkernel_Interpolate_NearestNaighbor
  #define CUDA_KERNEL_CALL(ID, TYPE)   \
    if (volumeInfo.InputDataType == ID) \
    CUDAkernel_renderAlgo_doIntegrationRender<TYPE, CUDAkernel_RayCastCompositeAlgorithm<TYPE, CUDAkernel_Interpolate_Trilinear<TYPE> > > <<< grid, threads >>>()

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
