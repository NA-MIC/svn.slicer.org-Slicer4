/*
 * Volume Rendering on CUDA.
 * Author: Nicholas Herlambang
 * 
 * NB. I haven't completed documentation in the source code, so it might be difficult to understand. I will add some comments later, or you can ask me at any time if you have any questions about this code or anything about CUDA. Please also modify Makefile for your environment. 
 *
 * I made this code simpler by disabling some optimization process such as the use of texture memory etc. Please also ask me at any time about this.
 */

#ifndef CUDA_RENDERALGO_H
#define CUDA_RENDERALGO_H
#include <vector_types.h>


/**
 * Execute volume rendering. There are also a lot of parameters here.
 */

void CUDArenderAlgo_doRender(uchar4* outputData, 
                             void* sourceData,
                             int inputDataType,
                             float* rotationMatrix, 
                             float* colorTransferFunction,
                             float* alphaTransferFunction,
                             float* minmax, float* lightVec, 
                             int sizeX, int sizeY, int sizeZ, 
                             int dsizeX, int dsizeY, 
                             float dispX, float dispY, float dispZ, 
                             float voxelSizeX, float voxelSizeY, float voxelSizeZ, 
                             int minThreshold, int maxThreshold, 
                             int sliceDistance);
#endif
