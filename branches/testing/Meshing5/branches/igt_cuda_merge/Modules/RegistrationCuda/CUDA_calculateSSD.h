#ifndef CUDA_CALCULATESSD_H
#define CUDA_CALCULATESSD_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// declaration, forward

extern "C"
float CUDAcalculateSSD_doCalculation(unsigned char* refData, unsigned char* tarData, int refSizeX, int refSizeY, int refSizeZ);

#endif
