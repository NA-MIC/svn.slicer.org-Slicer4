#ifndef __CUDA_RayCastCompositeAlgorithm_h__
#define __CUDA_RayCastCompositeAlgorithm_h__

#include "CUDA_renderRayCast.h"

template <typename T>
__device__ void CUDAkernel_RayCastCompositeAlgorithm(const int3& index,
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

            tempValue=((T*)volInfo.SourceData)[(int)(__float2int_rn(tempPos.z)*volInfo.VolumeSize.x*volInfo.VolumeSize.y + 
                __float2int_rn(tempPos.y)*volInfo.VolumeSize.x +
                __float2int_rn(tempPos.x))];
            /*interpolation start here*/
            //float posX = tempPos.x - __float2int_rd(tempPos.x);
            //float posY = tempPos.y - __float2int_rd(tempPos.y);
            //float posZ = tempPos.z - __float2int_rd(tempPos.z);

            //int base = __float2int_rd((tempPos.z)) * volInfo.VolumeSize.x*volInfo.VolumeSize.y + __float2int_rd((tempPos.y)) * volInfo.VolumeSize.x + __float2int_rd((tempPos.x));

            //tempValue=interpolate(posX, posY, posZ,
            //    ((T*)volInfo.SourceData)[base],
            //    ((T*)volInfo.SourceData)[(int)(base + volInfo.VolumeSize.x*volInfo.VolumeSize.y)],
            //    ((T*)volInfo.SourceData)[(int)(base + volInfo.VolumeSize.x)],
            //    ((T*)volInfo.SourceData)[(int)(base + volInfo.VolumeSize.x*volInfo.VolumeSize.y + volInfo.VolumeSize.x)],
            //    ((T*)volInfo.SourceData)[(int)(base + 1)],
            //    ((T*)volInfo.SourceData)[(int)(base + volInfo.VolumeSize.x*volInfo.VolumeSize.y + 1)],
            //    ((T*)volInfo.SourceData)[(int)(base + volInfo.VolumeSize.x + 1)],
            //    ((T*)volInfo.SourceData)[(int)(base + volInfo.VolumeSize.x*volInfo.VolumeSize.y + volInfo.VolumeSize.x + 1)]);
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

#endif /* __CUDA_RayCastCompositeAlgorithm_h__ */
