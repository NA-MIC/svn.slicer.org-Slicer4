#ifndef __CUDA_MATRIX_MATH_H__
#define __CUDA_MATRIX_MATH_H__

__device__ void MatMul(const float mat[4][4], float3* out, float inX, float inY, float inZ)
{
    out->x = mat[0][0] * inX + mat[0][1] * inY + mat[0][2] * inZ + mat[0][3] * 1.0;
    out->y = mat[1][0] * inX + mat[1][1] * inY + mat[1][2] * inZ + mat[1][3] * 1.0;
    out->z = mat[2][0] * inX + mat[2][1] * inY + mat[2][2] * inZ + mat[2][3] * 1.0;
}

__device__ void MatMul(const float mat[4][4], float3* out, const float3& in)
{
    out->x = mat[0][0] * in.x + mat[0][1] * in.y + mat[0][2] * in.z + mat[0][3] * 1.0;
    out->y = mat[1][0] * in.x + mat[1][1] * in.y + mat[1][2] * in.z + mat[1][3] * 1.0;
    out->z = mat[2][0] * in.x + mat[2][1] * in.y + mat[2][2] * in.z + mat[2][3] * 1.0;
}


__device__ float3 MatMul(const float mat[4][4], const float3& in)
{
    float3 out;
    MatMul(mat, &out, in);
    return out;
}

__device__ float3 MatMul(const float mat[4][4], float inX, float inY, float inZ)
{
    float3 out;
    MatMul(mat, &out, inX, inY, inZ);
    return out;
}

#endif /* __CUDA_MATRIX_MATH_H__ */
