#ifndef CUDA_CALCULATEMI_H
#define CUDA_CALCULATEMI_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// declaration, forward


extern "C"
double CUDAcalculateMI_doCalculation(unsigned char* refData, unsigned char* tarData, int refSizeX, int refSizeY, int refSizeZ, int tarSizeX, int tarSizeY, int tarSizeZ, float refThicknessX, float refThicknessY, float refThicknessZ, float tarThicknessX, float tarThicknessY, float tarThicknessZ);

#endif
