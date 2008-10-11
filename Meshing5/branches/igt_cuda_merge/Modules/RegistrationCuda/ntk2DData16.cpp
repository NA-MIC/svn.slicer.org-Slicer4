#include "ntk2DData16.h"
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

#define ACC2US(X,Y) ( (unsigned short*)m_buffer + ( (Y)*m_dataSize.x ) + (X) )

/**
 * ntk2DData16 class constructor
 */

ntk2DData16::ntk2DData16(ntkIntDimension dataSize)
{
  m_dataSize=dataSize;
  m_dataThickness.x=1.0;
  m_dataThickness.y=1.0;
  m_buffer=(unsigned short*)malloc(m_dataSize.x*m_dataSize.y*sizeof(unsigned short));
  m_bufferflag=1;
  m_dataType=NTK_UNSIGNED_CHAR;
  m_dimension=2;
}

ntk2DData16::ntk2DData16(ntkIntDimension dataSize, ntkFloatDimension dataThickness)
{
  m_dataSize=dataSize;
  m_dataThickness=dataThickness;
  m_buffer=(unsigned short*)malloc(m_dataSize.x*m_dataSize.y*sizeof(unsigned short));
  m_bufferflag=1;
  m_dataType=NTK_UNSIGNED_CHAR;
  m_dimension=2;
}

ntk2DData16::ntk2DData16(ntkIntDimension dataSize, unsigned short* buffer)
{
  m_dataSize=dataSize;
  m_dataThickness.x=1.0;
  m_dataThickness.y=1.0;
  m_buffer=buffer;
  m_bufferflag=0;
  m_dataType=NTK_UNSIGNED_CHAR;
  m_dimension=2;
}

ntk2DData16::ntk2DData16(ntkIntDimension dataSize, ntkFloatDimension dataThickness, unsigned short* buffer)
{
  m_dataSize=dataSize;
  m_dataThickness=dataThickness;
  m_buffer=buffer;
  m_bufferflag=0;
  m_dataType=NTK_UNSIGNED_CHAR;
  m_dimension=2;
}


/**
 * ntk2DData16 class destructor
 */

ntk2DData16::~ntk2DData16()
{
  if(m_buffer!=NULL && m_bufferflag!=0)
    free(m_buffer);
}

void ntk2DData16::readFile(const char* filename)
{
  fstream fp;
  fp.open(filename,ios::in);
  fp.read((char*)m_buffer, sizeof(unsigned short)*m_dataSize.x*m_dataSize.y);
  fp.close(); 
}

void ntk2DData16::writeFile(const char* filename)
{
  fstream fp;
  fp.open(filename,ios::out);
  fp.write((char*)m_buffer, sizeof(unsigned short)*m_dataSize.x*m_dataSize.y);
  fp.close();
}

void ntk2DData16::readBuffer(unsigned short* inBuffer)
{
  m_buffer=inBuffer;
}

void ntk2DData16::writeBuffer(unsigned short* outBuffer)
{
  memcpy(outBuffer, m_buffer, m_dataSize.x*m_dataSize.y*sizeof(unsigned short));
}

unsigned short* ntk2DData16::getBuffer()
{
  return (unsigned short*)m_buffer;
}

/**
 * Set the value of pixel (x,y)
 */

void ntk2DData16::setValue(ntkIntDimension dataPos, unsigned short inValue)
{
  *(ACC2US(dataPos.x,dataPos.y))=inValue;
}

/**
 * Get the value of pixel (x,y)
 */

unsigned short ntk2DData16::getValue(ntkIntDimension dataPos)
{
  return *(ACC2US(dataPos.x,dataPos.y));
}

