/*
 * Volume Rendering on CUDA.
 * Author: Nicholas Herlambang
 * 
 * 
 */

#ifndef CUDA_RENDERALGO_H
#define CUDA_RENDERALGO_H
#include <vector_types.h>


/**
 * Execute volume rendering. There are also a lot of parameters here.
 */

extern "C"
void CUDArenderAlgo_doRender(uchar4* outputData, 
                             void* sourceData,
                             int inputDataType,
                             float* rotationMatrix, 
                             float* colorTransferFunction,
                             float* alphaTransferFunction,
                             float* zBuffer,
                             float* minmax, float* lightVec, 
                             int sizeX, int sizeY, int sizeZ, 
                             int dsizeX, int dsizeY, 
                             float dispX, float dispY, float dispZ, 
                             float voxelSizeX, float voxelSizeY, float voxelSizeZ, 
                             int minThreshold, int maxThreshold, 
                             int sliceDistance);
#endif
