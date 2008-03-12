#ifndef NTKPROPERTIES_H
#define NTKPROPERTIES_H

#include <stdio.h>
#include <stdlib.h>

#ifdef WIN32

#include <fstream>
#include <iostream>
using namespace std;

#else

#include <fstream.h>
#include <iostream.h>

#endif

#ifndef M_PI
#define M_PI        3.14159265358979323846
#endif

typedef enum{NTK_CHAR,
         NTK_UNSIGNED_CHAR,
         NTK_SHORT,
         NTK_UNSIGNED_SHORT,
         NTK_INT,
         NTK_UNSIGNED_INT,
         NTK_LONG,
         NTK_UNSIGNED_LONG,
         NTK_FLOAT,
         NTK_DOUBLE}ntkDataType;

typedef enum{NTK_LITTLE_ENDIAN,
         NTK_BIG_ENDIAN}ntkDataOrder;
/*
typedef struct{
  char PatientID[256];
  char Date[256];
  char VolumeAttribute[256];
  int SamplesPerLine;
  int SampleOffset;
  int LinesPerFrame;
  int LineOffset;
  int FramesPerVolume;
  int FrameOffset;
  int ProbeOffset;
  int DiagnosticRange;
  int ElectricalAngle;
  int MechanicalAngle;
  char ScanDirection[256];
  int VolumeInterval;
  char CaptureMode[256];
}ntkUSDataProperties;
*/
typedef struct{
  bool active;
  int Rsmall;
  int Psmall;
  int Tsmall;
  float factor[8];
}ntkInterpolationProfile;

template <class Type = int>
struct ntkDimension{
  Type x;
  Type y;
  Type z;
  Type w;
  
  public: 
  ntkDimension<Type>(Type val1, Type val2, Type val3, Type val4){
    x=val1;
    y=val2;
    z=val3;
    w=val4;
  }

  ntkDimension<Type>(Type val1, Type val2, Type val3){
    x=val1;
    y=val2;
    z=val3;
    w=0;
  }

  ntkDimension<Type>(Type val1, Type val2){
    x=val1;
    y=val2;
    z=0;
    w=0;
  }

  ntkDimension<Type>(){
    x=0;
    y=0;
    z=0;
    w=0;
  }

  ntkDimension<Type> operator+(ntkDimension<Type> a){
    ntkDimension<Type> temp;
    temp.x=this->x+a.x;
    temp.y=this->y+a.y;
    temp.z=this->z+a.z;
    temp.w=this->w+a.w;
    
    return temp;
  }

  ntkDimension<Type> operator-(ntkDimension<Type> a){
    ntkDimension<Type> temp;
    temp.x=this->x-a.x;
    temp.y=this->y-a.y;
    temp.z=this->z-a.z;
    temp.w=this->w-a.w;
    
    return temp;
  }

  ntkDimension<Type> operator*(float a){
    ntkDimension<Type> temp;
    temp.x=this->x*a;
    temp.y=this->y*a;
    temp.z=this->z*a;
    temp.w=this->w*a;
    
    return temp;
  }

  
  ntkDimension<Type> operator/(float a){
    ntkDimension<Type> temp;
    temp.x=this->x/a;
    temp.y=this->y/a;
    temp.z=this->z/a;
    temp.w=this->w/a;
    
    return temp;
  }

  float dot(ntkDimension<Type> a){
    return this->x*a.x+this->y*a.y+this->z*a.z+this->w*a.w;
  }

  float abs(){
    return sqrt(this->x*this->x+this->y*this->y+this->z*this->z+this->w*this->w);
  }

  void normalize(){
    float abs = sqrt(this->x*this->x+this->y*this->y+this->z*this->z+this->w*this->w);
    if(abs!=0){
      float temp=1.0f/abs;
      this->x*=temp;
      this->y*=temp;
      this->z*=temp;
      this->w*=temp;
    }
  }
  
};

typedef ntkDimension<float> ntkFloatDimension;
typedef ntkDimension<int> ntkIntDimension;
typedef ntkDimension<double> ntkDoubleDimension;

template <class Type = unsigned char>
struct ntkColor{
  Type r;
  Type g;
  Type b;
  
  public: 
  ntkColor<Type>(Type val1, Type val2, Type val3){
    r=val1;
    g=val2;
    b=val3;
  }

  ntkColor<Type>(){
    r=0;
    g=0;
    b=0;
  }
};

typedef ntkColor<unsigned char> ntkUnsignedCharColor;
typedef ntkColor<unsigned short> ntkUnsignedShortColor;

typedef unsigned char tSourceData;

typedef struct{
  unsigned char dummy; // -> sizeof(tColor) == 4, fits better to cache...
  unsigned char r;
  unsigned char g;
  unsigned char b;
} tColor;

typedef struct{
  unsigned char r;
  unsigned char g;
  unsigned char b;
} tCompressedColor;


/*typedef struct{
  float x;
  float y;
} tLensCoord;
  */

/*
  stores the center of lense (relative to the center of the display)
  of one pixel in the lensarray
  and the direction of the ray (vz is always 1.0)
*/
typedef struct{
  float x;
  float y;
  float vx;
  float vy;
} tLensMap;


//typedef float *tMatrix;

typedef float tHomMatrix[4][4];

typedef struct {
  float x,y,z;
}tVector;

typedef struct {
  int x,y;
}tiVector2;

typedef struct {
  int x,y,z;
}tiVector3;

const float PI = 3.14159265;

#define SQR(a) ((a)*(a))

#endif
