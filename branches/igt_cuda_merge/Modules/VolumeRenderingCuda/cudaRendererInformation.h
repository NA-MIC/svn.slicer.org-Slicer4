#ifndef __CUDARENDERERINFORMATION_H__
#define __CUDARENDERERINFORMATION_H__

#include "vector_types.h"

//! A Datastucture located on the cuda hardware that holds all the information about the renderer.
extern "C"
typedef struct
{
    //! The resolution of the rendering screen.
    uint2          Resolution;
    //! The color depth of the rendering screen.
    unsigned int   ColorDepth;

    //! Count of lights in the scene
    unsigned int   LightCount;
    //! The vectors to the lights
    float3*        LightVectors;

    float3         CameraPos;
    float3         CameraDirection;
    float3         ViewUp;
    float3         HorizontalVec;  // Horizontal Vector
    float3         VerticalVec;    // Vertical Vector

    uchar4*        OutputImage;
    float*         ZBuffer;
    float2         ClippingRange;
} cudaRendererInformation;

#endif /* __CUDARENDERERINFORMATION_H__ */
