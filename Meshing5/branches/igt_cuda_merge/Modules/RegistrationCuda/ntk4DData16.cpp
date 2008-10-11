#include "ntk4DData16.h"
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

#define ACC4US(X,Y,Z,W) ( (unsigned short*)m_buffer + ( (W)*m_dataSize.x*m_dataSize.y*m_dataSize.z ) + ( (Z)*m_dataSize.x*m_dataSize.y ) + ( (Y)*m_dataSize.x ) + (X) )

/**
 * ntk4DData16 class constructor
 */

ntk4DData16::ntk4DData16(ntkIntDimension dataSize)
{
  m_dataSize=dataSize;
  m_dataThickness.x=1.0;
  m_dataThickness.y=1.0;
  m_dataThickness.z=1.0;
  m_dataThickness.w=1.0;
  m_buffer=(unsigned short*)malloc(m_dataSize.x*m_dataSize.y*m_dataSize.z*m_dataSize.w*sizeof(unsigned short));
  m_bufferflag=1;
  m_dataType=NTK_UNSIGNED_CHAR;
  m_dimension=4;
}

ntk4DData16::ntk4DData16(ntkIntDimension dataSize, ntkFloatDimension dataThickness)
{
  m_dataSize=dataSize;
  m_dataThickness=dataThickness;
  m_buffer=(unsigned short*)malloc(m_dataSize.x*m_dataSize.y*m_dataSize.z*m_dataSize.w*sizeof(unsigned short));
  m_bufferflag=1;
  m_dataType=NTK_UNSIGNED_CHAR;
  m_dimension=4;
}

ntk4DData16::ntk4DData16(ntkIntDimension dataSize, unsigned short* buffer)
{
  m_dataSize=dataSize;
  m_dataThickness.x=1.0;
  m_dataThickness.y=1.0;
  m_dataThickness.z=1.0;
  m_dataThickness.w=1.0;
  m_buffer=buffer;
  m_bufferflag=0;
  m_dataType=NTK_UNSIGNED_CHAR;
  m_dimension=4;
}


ntk4DData16::ntk4DData16(ntkIntDimension dataSize, ntkFloatDimension dataThickness, unsigned short* buffer)
{
  m_dataSize=dataSize;
  m_dataThickness=dataThickness;
  m_buffer=buffer;
  m_bufferflag=0;
  m_dataType=NTK_UNSIGNED_CHAR;
  m_dimension=4;
}

/**
 * ntk4DData16 class destructor
 */

ntk4DData16::~ntk4DData16()
{
  if(m_buffer!=NULL && m_bufferflag!=0)
    free(m_buffer);
}

void ntk4DData16::readFile(const char* filename)
{
  fstream fp;
  fp.open(filename,ios::in);
  fp.read((char*)m_buffer, sizeof(unsigned short)*m_dataSize.x*m_dataSize.y*m_dataSize.z*m_dataSize.w);
  fp.close();
}

void ntk4DData16::writeFile(const char* filename)
{
  fstream fp;
  fp.open(filename,ios::out);
  fp.write((char*)m_buffer, sizeof(unsigned short)*m_dataSize.x*m_dataSize.y*m_dataSize.z*m_dataSize.w);
  fp.close();
}

void ntk4DData16::readBuffer(unsigned short* inBuffer)
{
  m_buffer=inBuffer;
}

void ntk4DData16::writeBuffer(unsigned short* outBuffer)
{
  memcpy(outBuffer, m_buffer, m_dataSize.x*m_dataSize.y*m_dataSize.z*m_dataSize.w*sizeof(unsigned short));
}

unsigned short* ntk4DData16::getBuffer()
{
  return (unsigned short*)m_buffer;
}

/**
 * Set the value of voxel (x,y,z,w)
 */

void ntk4DData16::setValue(ntkIntDimension dataPos, unsigned short inValue)
{
  *(ACC4US(dataPos.x,dataPos.y,dataPos.z,dataPos.w))=inValue;
}

/**
 * Get the value of voxel (x,y,z,w)
 */

unsigned short ntk4DData16::getValue(ntkIntDimension dataPos)
{
  return *(ACC4US(dataPos.x,dataPos.y,dataPos.z,dataPos.w));
}

