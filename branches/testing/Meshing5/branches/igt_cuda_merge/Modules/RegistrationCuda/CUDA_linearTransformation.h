#ifndef CUDA_LINEARTRANSFORMATION_H
#define CUDA_LINEARTRANSFORMATION_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// declaration, forward

extern "C"
void CUDAlinearTransformation_doTransformation(unsigned char* inputData, unsigned char* outputData, float* inverseMatrix, int inSizeX, int inSizeY, int inSizeZ, int outSizeX, int outSizeY, int outSizeZ, float inThicknessX, float inThicknessY, float inThicknessZ, float outThicknessX, float outThicknessY, float outThicknessZ);

#endif
