#ifndef __CUDA_RayCastCompositeAlgorithm_h__
#define __CUDA_RayCastCompositeAlgorithm_h__

#include "CUDA_renderRayCast.h"

template <typename T> class CUDAkernel_Interpolate_NearestNaighbor
{
public:
    __device__ float operator()(const void* SourceData, const int3& VolumeSize, const float3& Pos)
    {
        return ((T*)SourceData)[(int)(__float2int_rn(Pos.z)*VolumeSize.x*VolumeSize.y + 
            __float2int_rn(Pos.y)*VolumeSize.x +
            __float2int_rn(Pos.x))];
    }
};

template <typename T> class CUDAkernel_Interpolate_Trilinear
{
public:
    __device__ float operator()(const void* SourceData, const int3& VolumeSize, const float3& Pos)
    {
        float posX = Pos.x - __float2int_rd(Pos.x);
        float posY = Pos.y - __float2int_rd(Pos.y);
        float posZ = Pos.z - __float2int_rd(Pos.z);

        int base = __float2int_rd((Pos.z)) * VolumeSize.x * VolumeSize.y + 
            __float2int_rd((Pos.y)) * VolumeSize.x +
            __float2int_rd((Pos.x));

        return interpolate(posX, posY, posZ,
            ((T*)SourceData)[base],
            ((T*)SourceData)[(int)(base + VolumeSize.x * VolumeSize.y)],
            ((T*)SourceData)[(int)(base + VolumeSize.x)],
            ((T*)SourceData)[(int)(base + VolumeSize.x * VolumeSize.y + VolumeSize.x)],
            ((T*)SourceData)[(int)(base + 1)],
            ((T*)SourceData)[(int)(base + VolumeSize.x * VolumeSize.y + 1)],
            ((T*)SourceData)[(int)(base + VolumeSize.x + 1)],
            ((T*)SourceData)[(int)(base + VolumeSize.x * VolumeSize.y + VolumeSize.x + 1)]);
    }
};


template <class T, template<class T> class InterpolationMethod> class CUDAkernel_RayCastCompositeAlgorithm
{
public:
    __device__ void operator()(const int3& index,
        int outindex,
        const float* minmax /*[6] */,
        const float3* rayMap,
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
        float initialZBuffer  =  zBuffer[index.z]; //!< initial zBuffer from input

        float A = renInfo.ClippingRange.y / (renInfo.ClippingRange.y - renInfo.ClippingRange.x);
        float B = renInfo.ClippingRange.y * renInfo.ClippingRange.x / (renInfo.ClippingRange.x - renInfo.ClippingRange.y);

        //perform ray tracing until integration of alpha value reach threshold 
        while(depth < initialZBuffer) {
            distFromCam = B / ( depth - A);

            //calculate current position in ray tracing
            MatMul(volInfo.Transform, &tempPos, 
                (renInfo.CameraPos.x + distFromCam * rayMap[index.z*2+1].x),
                (renInfo.CameraPos.y + distFromCam * rayMap[index.z*2+1].y),
                (renInfo.CameraPos.z + distFromCam * rayMap[index.z*2+1].z));

            // if current position is in ROI
            if(tempPos.x >= minmax[0] && tempPos.x < minmax[1] &&
                tempPos.y >= minmax[2] && tempPos.y < minmax[3] &&
                tempPos.z >= minmax[4] && tempPos.z < minmax[5] )
            {
                //check whether current position is in front of z buffer wall

                InterpolationMethod<T> interpolate;
                tempValue = interpolate(volInfo.SourceData, volInfo.VolumeSize, tempPos);
                /*interpolation start here*/

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
            depth += 1.0/256.0 * volInfo.SampleDistance;
        }
    }
};
#endif /* __CUDA_RayCastCompositeAlgorithm_h__ */
