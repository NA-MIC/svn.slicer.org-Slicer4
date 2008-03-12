#include "ntk2DData.h"
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

#define ACC2UC(X,Y) ( (unsigned char*)m_buffer + ( (Y)*m_dataSize.x ) + (X) )

/**
 * ntk2DData class constructor
 */

ntk2DData::ntk2DData(ntkIntDimension dataSize)
{
  m_dataSize=dataSize;
  m_dataThickness.x=1.0;
  m_dataThickness.y=1.0;
  m_buffer=(unsigned char*)malloc(m_dataSize.x*m_dataSize.y*sizeof(unsigned char));
  int i;
  for(i=0;i<dataSize.x*dataSize.y;i++){
    *((unsigned char*)m_buffer+i)=0;
  }

  m_bufferflag=1;
  m_dataType=NTK_UNSIGNED_CHAR;
  m_dimension=2;
}

ntk2DData::ntk2DData(ntkIntDimension dataSize, ntkFloatDimension dataThickness)
{
  m_dataSize=dataSize;
  m_dataThickness=dataThickness;
  m_buffer=(unsigned char*)malloc(m_dataSize.x*m_dataSize.y*sizeof(unsigned char));
  int i;
  for(i=0;i<dataSize.x*dataSize.y;i++){
    *((unsigned char*)m_buffer+i)=0;
  }
  m_bufferflag=1;
  m_dataType=NTK_UNSIGNED_CHAR;
  m_dimension=2;
}

ntk2DData::ntk2DData(ntkIntDimension dataSize, unsigned char* buffer)
{
  m_dataSize=dataSize;
  m_dataThickness.x=1.0;
  m_dataThickness.y=1.0;
  m_buffer=buffer;
  m_bufferflag=0;
  m_dataType=NTK_UNSIGNED_CHAR;
  m_dimension=2;
}

ntk2DData::ntk2DData(ntkIntDimension dataSize, ntkFloatDimension dataThickness, unsigned char* buffer)
{
  m_dataSize=dataSize;
  m_dataThickness=dataThickness;
  m_buffer=buffer;
  m_bufferflag=0;
  m_dataType=NTK_UNSIGNED_CHAR;
  m_dimension=2;
}


/**
 * ntk2DData class destructor
 */

ntk2DData::~ntk2DData()
{
  if(m_buffer!=NULL && m_bufferflag!=0)
    free(m_buffer);
}

void ntk2DData::readFile(const char* filename)
{
  fstream fp;
  fp.open(filename,ios::in);
  fp.read((char*)m_buffer, sizeof(unsigned char)*m_dataSize.x*m_dataSize.y);
  fp.close();
}

void ntk2DData::writeFile(const char* filename)
{
  fstream fp;
  fp.open(filename,ios::out);
  fp.write((char*)m_buffer, sizeof(unsigned char)*m_dataSize.x*m_dataSize.y);
  fp.close();
}

void ntk2DData::readBuffer(unsigned char* inBuffer)
{
  m_buffer=inBuffer;
}

void ntk2DData::writeBuffer(unsigned char* outBuffer)
{
  memcpy(outBuffer, m_buffer, m_dataSize.x*m_dataSize.y*sizeof(unsigned char));
}

unsigned char* ntk2DData::getBuffer()
{
  return (unsigned char*)m_buffer;
}

/**
 * Set the value of pixel (x,y)
 */

void ntk2DData::setValue(ntkIntDimension dataPos, unsigned char inValue)
{
  *(ACC2UC(dataPos.x,dataPos.y))=inValue;
}

/**
 * Get the value of pixel (x,y)
 */

unsigned char ntk2DData::getValue(ntkIntDimension dataPos)
{
  return *(ACC2UC(dataPos.x,dataPos.y));
}

