#ifndef __CUDAVOLUMEINFORMATION_H__
#define __CUDAVOLUMEINFORMATION_H__

#include "vector_types.h"

//! A datastructure located on the cudacard that holds the information of the volume.
extern "C"
typedef struct __align__(16) {
    void*           SourceData;
    int             InputDataType;
    
    // The size of the volume
    int3            VolumeSize;
    float           Transform[4][4];


    //! The Color Transfer Function with a size of FunctionSize * 3 (rgb)
    float*          ColorTransferFunction;
    //! The Alpha Transfer Function with a size of FunctionSize
    float*          AlphaTransferFunction;
    //! The Size of the above Function
    unsigned int    FunctionSize;
    float           FunctionRange[2];

    float           Ambient;        //!< Ambient color part
    float           Diffuse;        //!< Diffuse color part
    float           Specular;       //!< Specular color part
    float           SpecularPower;  //!< The power of the specular color

    //! The minimum and Maximum Values of the Volume
    float           MinMaxValue[6];

    //! The minimal Threshold of the Input Color Value     
    int             MinThreshold;
    //! The Maximum Threshold of the Input Color Value
    int             MaxThreshold;

    //! The stepping accuracy to raster along the ray.
    float           SampleDistance;

} cudaVolumeInformation;
#endif /* __CUDAVOLUMEINFORMATION_H__ */
