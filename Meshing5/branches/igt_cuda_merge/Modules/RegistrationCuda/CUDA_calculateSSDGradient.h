#ifndef CUDA_CALCULATESSDGRADIENT_H
#define CUDA_CALCULATESSDGRADIENT_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// declaration, forward

extern "C"
float CUDAcalculateSSDGradient_doCalculation(unsigned char* reference, unsigned char* target, float* transSpline, float* SSDgradient, int inSizeX, int inSizeY, int inSizeZ, int splineLevel);

#endif
