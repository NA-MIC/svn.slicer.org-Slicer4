#ifndef __CUDARENDERERINFORMATION_H__
#define __CUDARENDERERINFORMATION_H__

#include "cuda_runtime_api.h"

//! A Datastucture located on the cuda hardware that holds all the information about the renderer.
extern "C"
typedef struct
{
    //! The resolution of the rendering screen.
    unsigned int   ResolutionX, ResolutionY;
    //! The color depth of the rendering screen.
    unsigned int   ColorDepth;

    //! Count of lights in the scene
    unsigned int   LightCount;
    //! The vectors to the lights
    float3*        LightVectors;

    float          CameraPosX, CameraPosY, CameraPosZ;
    float          TargetPosX, TargetPosY, TargetPosZ;
    float          UpX, UpY, UpZ;

    float*         ZBuffer;
    float          NearPlane;
    float          FarPlane;

} cudaRendererInformation;

#endif /* __CUDARENDERERINFORMATION_H__ */
