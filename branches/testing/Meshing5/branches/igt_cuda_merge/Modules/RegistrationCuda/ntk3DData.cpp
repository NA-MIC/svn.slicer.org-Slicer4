#include "ntk3DData.h"
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

#define ACC3UC(X,Y,Z) ( (unsigned char*)m_buffer + ( (Z)*m_dataSize.x*m_dataSize.y ) + ( (Y)*m_dataSize.x ) + (X) )

/**
 * ntk3DData class constructor
 */

ntk3DData::ntk3DData(ntkIntDimension dataSize)
{
  int i,j,k;

  m_dataSize=dataSize;
  m_dataThickness.x=1.0;
  m_dataThickness.y=1.0;
  m_dataThickness.z=1.0;
  m_buffer=(unsigned char*)malloc(m_dataSize.x*m_dataSize.y*m_dataSize.z*sizeof(unsigned char));
  m_bufferflag=1;
  m_dataType=NTK_UNSIGNED_CHAR;
  m_dimension=3;
  m_lockBuffer=true;

  for(i=0;i<m_dataSize.x;i++){
    for(j=0;j<m_dataSize.y;j++){
      for(k=0;k<m_dataSize.z;k++){
    *((unsigned char*)m_buffer+k*m_dataSize.x*m_dataSize.y+j*m_dataSize.x+i)=0;
      }
    }
  }
  
   
}

ntk3DData::ntk3DData(ntkIntDimension dataSize, unsigned char initVal)
{
  int i,j,k;
  
  m_dataSize=dataSize;
  m_dataThickness.x=1.0;
  m_dataThickness.y=1.0;
  m_dataThickness.z=1.0;
  m_buffer=(unsigned char*)malloc(m_dataSize.x*m_dataSize.y*m_dataSize.z*sizeof(unsigned char));
  m_bufferflag=1;
  m_dataType=NTK_UNSIGNED_CHAR;
  m_dimension=3;
  m_lockBuffer=true;
  
  for(i=0;i<m_dataSize.x;i++){
    for(j=0;j<m_dataSize.y;j++){
      for(k=0;k<m_dataSize.z;k++){
    *((unsigned char*)m_buffer+k*m_dataSize.x*m_dataSize.y+j*m_dataSize.x+i)=initVal;
      }
    }
  }
}

ntk3DData::ntk3DData(ntkIntDimension dataSize, ntkFloatDimension dataThickness)
{
  int i,j,k;

  m_dataSize=dataSize;
  m_dataThickness=dataThickness;
  m_buffer=(unsigned char*)malloc(m_dataSize.x*m_dataSize.y*m_dataSize.z*sizeof(unsigned char));
  m_bufferflag=1;
  m_dataType=NTK_UNSIGNED_CHAR;
  m_dimension=3;
  m_lockBuffer=true;

  for(i=0;i<m_dataSize.x;i++){
    for(j=0;j<m_dataSize.y;j++){
      for(k=0;k<m_dataSize.z;k++){
    *((unsigned char*)m_buffer+k*m_dataSize.x*m_dataSize.y+j*m_dataSize.x+i)=0;
      }
    }
  }
}

ntk3DData::ntk3DData(ntkIntDimension dataSize, unsigned char* buffer)
{
  m_dataSize=dataSize;
  m_dataThickness.x=1.0;
  m_dataThickness.y=1.0;
  m_dataThickness.z=1.0;
  m_buffer=buffer;
  m_bufferflag=0;
  m_dataType=NTK_UNSIGNED_CHAR;
  m_dimension=3;
  m_lockBuffer=true;
}

ntk3DData::ntk3DData(ntkIntDimension dataSize, ntkFloatDimension dataThickness, unsigned char* buffer)
{
  m_dataSize=dataSize;
  m_dataThickness=dataThickness;
  m_buffer=buffer;
  m_bufferflag=0;
  m_dataType=NTK_UNSIGNED_CHAR;
  m_dimension=3;
  m_lockBuffer=true;
}

/**
 * ntk3DData class destructor
 */

ntk3DData::~ntk3DData()
{
  if(m_buffer!=NULL && m_lockBuffer)
    free(m_buffer);
}

void ntk3DData::readFile(const char* filename)
{
  fstream fp;
  fp.open(filename,ios::in);
  fp.read((char*)m_buffer, sizeof(unsigned char)*m_dataSize.x*m_dataSize.y*m_dataSize.z);
  fp.close();
}

void ntk3DData::writeFile(const char* filename)
{
  fstream fp;
  fp.open(filename,ios::out);
  fp.write((char*)m_buffer, sizeof(unsigned char)*m_dataSize.x*m_dataSize.y*m_dataSize.z);
  fp.close();
}

void ntk3DData::readNTKFile(const char* filename)
{
  
}

void ntk3DData::writeNTKFile(const char* filename)
{
  fstream fp;
  fp.open(filename,ios::out);
  unsigned char* buffer=(unsigned char*)malloc((m_dataSize.x*m_dataSize.y*m_dataSize.z+50)*sizeof(unsigned char));

  int suc=sizeof(unsigned char);
  int si=sizeof(int);
  int sd=sizeof(double);

  *(buffer)=1;
  *(int*)(buffer+suc)=m_dataSize.x;
  *(int*)(buffer+suc+si)=m_dataSize.y;
  *(int*)(buffer+suc+si*2)=m_dataSize.z;
  *(double*)(buffer+suc+si*3)=m_dataThickness.x;
  *(double*)(buffer+suc+si*3+sd)=m_dataThickness.y;
  *(double*)(buffer+suc+si*3+sd*2)=m_dataThickness.z;

  memcpy(buffer+50, m_buffer, m_dataSize.x*m_dataSize.y*m_dataSize.z*suc);
  
  fp.write((char*)buffer, sizeof(unsigned char)*m_dataSize.x*m_dataSize.y*m_dataSize.z+ 50);

  free(buffer);
  fp.close();
}

void ntk3DData::readBuffer(unsigned char* inBuffer)
{
  m_buffer=inBuffer;
}

void ntk3DData::writeBuffer(unsigned char* outBuffer)
{
  memcpy(outBuffer, m_buffer, m_dataSize.x*m_dataSize.y*m_dataSize.z*sizeof(unsigned char));
}

unsigned char* ntk3DData::getBuffer()
{
  return (unsigned char*)m_buffer;
}

/**
 * Set the value of voxel (x,y,z)
 */

void ntk3DData::setValue(ntkIntDimension dataPos, unsigned char inValue)
{
  *(ACC3UC(dataPos.x,dataPos.y,dataPos.z))=inValue;
}

/**
 * Get the value of voxel (x,y,z)
 */

unsigned char ntk3DData::getValue(ntkIntDimension dataPos)
{
  return *(ACC3UC(dataPos.x,dataPos.y,dataPos.z));
}

void ntk3DData::setDataContrast(double center, double window){
  if(center<0 || center>255 || window<0){
    printf("NTK error: cannot set contrast. parameter error");  
  }else{
    
  }
}

ntk3DData* ntk3DData::getDataWithNewContrast(double center, double window){
    return NULL;
}

void ntk3DData::lockBuffer(bool lock){
  m_lockBuffer=lock;
}

void ntk3DData::flipXAxis(){
  unsigned char temp;
  int i,j,k;
  
  for(i=0;i<m_dataSize.x;i++){
    for(j=0;j<m_dataSize.y/2;j++){
      for(k=0;k<m_dataSize.z;k++){
    temp=*((unsigned char*)m_buffer+k*m_dataSize.x*m_dataSize.y+j*m_dataSize.x+i);
    *((unsigned char*)m_buffer+k*m_dataSize.x*m_dataSize.y+j*m_dataSize.x+i)=*((unsigned char*)m_buffer+k*m_dataSize.x*m_dataSize.y+j*m_dataSize.x+(m_dataSize.x-1-i));
    *((unsigned char*)m_buffer+k*m_dataSize.x*m_dataSize.y+j*m_dataSize.x+(m_dataSize.x-1-i))=temp;
      }
    }
  }
}
