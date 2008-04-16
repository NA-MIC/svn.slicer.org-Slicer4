#ifndef __CUDA_RayCastCompositeAlgorithm_h__
#define __CUDA_RayCastCompositeAlgorithm_h__

#include "CUDA_matrix_math.h"
#include "CUDA_interpolation.h"

template <class T, template<class T> class InterpolationMethod> class CUDAkernel_RayCastCompositeAlgorithm
{
public:
    __device__ void operator()(const int3& index,
        int outindex,
        const float* minmax /*[6] */,
        const float2* minmaxTrace,
        const float2& ZBufferFactors,
        const float3* rayMap,
        const cudaVolumeInformation& volInfo,
        const cudaRendererInformation& renInfo,
        float3* outputVal,
        float* zBuffer,
        float* remainingOpacity)
    {
        //ray tracing start from here

        float3 tempPos;     //!< variables to store current position
        T tempValue;        //!< A Temporary color value
        int tempIndex;      //!< Temporaty index in the 3D data
        float alpha;        //!< Alpha value of current voxel

        float depth = minmaxTrace[index.z].x;  //current step distance from camera

        float3 camPos = MatMul(volInfo.Transform, renInfo.CameraPos);

        //perform ray tracing until integration of alpha value reach threshold 
        while(depth < minmaxTrace[index.z].y) {
            tempPos.x = ( rayMap[index.z*2].x + depth * rayMap[index.z*2+1].x);
            tempPos.y = ( rayMap[index.z*2].y + depth * rayMap[index.z*2+1].y);
            tempPos.z = ( rayMap[index.z*2].z + depth * rayMap[index.z*2+1].z);

            //check whether current position is in front of z buffer wall
            float3 dist = make_float3(tempPos.x - camPos.x, tempPos.y - camPos.y, tempPos.z - camPos.z);
            if (ZBufferFactors.x + ZBufferFactors.y / VecLen(dist) > zBuffer[index.z])
                break;

            // if current position is in ROI
            if(tempPos.x >= minmax[0] && tempPos.x < minmax[1] &&
                tempPos.y >= minmax[2] && tempPos.y < minmax[3] &&
                tempPos.z >= minmax[4] && tempPos.z < minmax[5] )
            {

                // Interpolation 
                InterpolationMethod<T> interpolate;
                tempValue = interpolate(volInfo.SourceData, volInfo.VolumeSize, tempPos);

                tempIndex = __float2int_rn((volInfo.FunctionSize-1) * (float)(tempValue-volInfo.FunctionRange[0]) /
                    (float)(volInfo.FunctionRange[1] - volInfo.FunctionRange[0]));
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
                        zBuffer[index.z] = 1/depth;
                        break;
                    }
                }
            }
            depth += 1/*(minmaxTrace[index.z].y - minmaxTrace[index.z].x)*//256.0; 
        }
    }
};
#endif /* __CUDA_RayCastCompositeAlgorithm_h__ */
