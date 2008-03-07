#ifndef __CUDA_RayCastAlgorithm_h__
#define __CUDA_RayCastAlgorithm_h__

__device__ void MatMul(const float mat[4][4], float3* out, float inX, float inY, float inZ);

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

#endif /* __CUDA_RayCastAlgorithm_h__ */
