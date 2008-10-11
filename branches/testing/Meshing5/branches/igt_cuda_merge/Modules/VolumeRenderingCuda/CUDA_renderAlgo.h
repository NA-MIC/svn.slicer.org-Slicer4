/*
 * Volume Rendering on CUDA.
 * Author: Nicholas Herlambang
 * Second Author: Benjamin Grauer
 */

#ifndef CUDA_RENDERALGO_H
#define CUDA_RENDERALGO_H
#include "cudaRendererInformation.h"
#include "cudaVolumeInformation.h"

/**
 * Execute volume rendering. There are also a lot of parameters here.
 */
extern "C"
void CUDArenderAlgo_doRender(const cudaRendererInformation& renderInfo,
                             const cudaVolumeInformation& volumeInfo);
#endif
