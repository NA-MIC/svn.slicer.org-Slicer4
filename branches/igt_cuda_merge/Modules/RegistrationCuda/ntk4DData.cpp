#include "ntk4DData.h"
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

#define ACC4UC(X,Y,Z,W) ( (unsigned char*)m_buffer + ( (W)*m_dataSize.x*m_dataSize.y*m_dataSize.z ) + ( (Z)*m_dataSize.x*m_dataSize.y ) + ( (Y)*m_dataSize.x ) + (X) )

/**
 * ntk4DData class constructor
 */

ntk4DData::ntk4DData(ntkIntDimension dataSize)
{
  m_dataSize=dataSize;
  m_dataThickness.x=1.0;
  m_dataThickness.y=1.0;
  m_dataThickness.z=1.0;
  m_dataThickness.w=1.0;
  m_buffer=(unsigned char*)malloc(m_dataSize.x*m_dataSize.y*m_dataSize.z*m_dataSize.w*sizeof(unsigned char));
  m_bufferflag=1;
  m_dataType=NTK_UNSIGNED_CHAR;
  m_dimension=4;
}

ntk4DData::ntk4DData(ntkIntDimension dataSize, ntkFloatDimension dataThickness)
{
  m_dataSize=dataSize;
  m_dataThickness=dataThickness;
  m_buffer=(unsigned char*)malloc(m_dataSize.x*m_dataSize.y*m_dataSize.z*m_dataSize.w*sizeof(unsigned char));
  m_bufferflag=1;
  m_dataType=NTK_UNSIGNED_CHAR;
  m_dimension=4;
}

ntk4DData::ntk4DData(ntkIntDimension dataSize, unsigned char* buffer)
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


ntk4DData::ntk4DData(ntkIntDimension dataSize, ntkFloatDimension dataThickness, unsigned char* buffer)
{
  m_dataSize=dataSize;
  m_dataThickness=dataThickness;
  m_buffer=buffer;
  m_bufferflag=0;
  m_dataType=NTK_UNSIGNED_CHAR;
  m_dimension=4;
}

/**
 * ntk4DData class destructor
 */

ntk4DData::~ntk4DData()
{
  if(m_buffer!=NULL && m_bufferflag!=0)
    free(m_buffer);
}

void ntk4DData::readFile(const char* filename)
{
  fstream fp;
  fp.open(filename,ios::in);
  fp.read((char*)m_buffer, sizeof(unsigned char)*m_dataSize.x*m_dataSize.y*m_dataSize.z*m_dataSize.w);
  fp.close();
}

void ntk4DData::writeFile(const char* filename)
{
  fstream fp;
  fp.open(filename,ios::out);
  fp.write((char*)m_buffer, sizeof(unsigned char)*m_dataSize.x*m_dataSize.y*m_dataSize.z*m_dataSize.w);
  fp.close();
}

void ntk4DData::readBuffer(unsigned char* inBuffer)
{
  m_buffer=inBuffer;
}

void ntk4DData::writeBuffer(unsigned char* outBuffer)
{
  memcpy(outBuffer, m_buffer, m_dataSize.x*m_dataSize.y*m_dataSize.z*m_dataSize.w*sizeof(unsigned char));
}

unsigned char* ntk4DData::getBuffer()
{
  return (unsigned char*)m_buffer;
}

/**
 * Set the value of voxel (x,y,z,w)
 */

void ntk4DData::setValue(ntkIntDimension dataPos, unsigned char inValue)
{
  *(ACC4UC(dataPos.x,dataPos.y,dataPos.z,dataPos.w))=inValue;
}

/**
 * Get the value of voxel (x,y,z,w)
 */

unsigned char ntk4DData::getValue(ntkIntDimension dataPos)
{
  return *(ACC4UC(dataPos.x,dataPos.y,dataPos.z,dataPos.w));
}

