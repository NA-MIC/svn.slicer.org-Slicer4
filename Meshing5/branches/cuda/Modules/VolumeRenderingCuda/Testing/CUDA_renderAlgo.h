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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cutil.h>

/**
 * Initialization. This function prepare GPU memory for rendering 3D data of size (sizeX, sizeY, sizeZ) on display size of (dsizeX, dsizeY)
 */

void CUDArenderAlgo_init(int sizeX, int sizeY, int sizeZ, int dsizeX, int dsizeY);

/**
 * Load data from CPU into GPU. There are a lot of parameters here. Please ask me for more details
 */

void CUDArenderAlgo_loadData(unsigned char* sourceData, int sizeX, int sizeY, int sizeZ);

/**
 * Execute volume rendering. There are also a lot of parameters here.
 */

void CUDArenderAlgo_doRender(float* rotationMatrix, float* color, float* minmax, float* lightVec, int sizeX, int sizeY, int sizeZ, int dsizeX, int dsizeY, float dispX, float dispY, float dispZ, float voxelSizeX, float voxelSizeY, float voxelSizeZ, int minThreshold, int maxThreshold, int sliceDistance);

/**
 * Copy the result from GPU memory to CPU memory. The resulted image is RGBA image of size (dsizeX, dsizeY)
 */

void CUDArenderAlgo_getResult(unsigned char** resultImagePointer, int dsizeX, int dsizeY);

/**
 * Free GPU memories.
 */

void CUDArenderAlgo_delete();

#endif
