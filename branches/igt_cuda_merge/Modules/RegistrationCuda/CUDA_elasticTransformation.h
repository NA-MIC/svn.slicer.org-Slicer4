#ifndef CUDA_ELASTICTRANSFORMATION_H
#define CUDA_ELASTICTRANSFORMATION_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// declaration, forward

void CUDAelasticTransformation_doTransformation(unsigned char* inputData, unsigned char* outputData, float* transSpline, int inSizeX, int inSizeY, int inSizeZ, int splineLevel);

#endif
