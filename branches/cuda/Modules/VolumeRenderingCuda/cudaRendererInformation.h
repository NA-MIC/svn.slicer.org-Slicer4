#ifndef __CUDARENDERERINFORMATION_H__
#define __CUDARENDERERINFORMATION_H__

#include "cuda_runtime_api.h"

//! A Datastucture located on the cuda hardware that holds all the information about the renderer.
extern "C"
typedef struct
{
    //! The resolution of the rendering screen.
    unsigned int   Resolution[2];
    //! The color depth of the rendering screen.
    unsigned int   ColorDepth;

    //! Count of lights in the scene
    unsigned int   LightCount;
    //! The vectors to the lights
    float3*        LightVectors;

    float          CameraPos[3];
    float          TargetPos[3];
    float          ViewUp[3];

    float*         ZBuffer;
    float          NearPlane;
    float          FarPlane;
} cudaRendererInformation;

#endif /* __CUDARENDERERINFORMATION_H__ */
