/*
 * Volume Rendering on CUDA.
 * Author: Nicholas Herlambang
 * 
 * 
 */

#ifndef CUDA_RENDERALGO_H
#define CUDA_RENDERALGO_H
#include <vector_types.h>
#include "cudaRendererInformation.h"
#include "cudaVolumeInformation.h"

/**
 * Execute volume rendering. There are also a lot of parameters here.
 */

extern "C"
void CUDArenderAlgo_doRender(uchar4* outputData, 
                             const cudaRendererInformation& renderInfo,
                             const cudaVolumeInformation& volumeInfo);
#endif
