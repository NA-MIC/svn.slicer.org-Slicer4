#ifndef NTKDATA_H
#define NTKDATA_H

#include <string.h>
#include "ntkProperties.h"

/**
 * Data Class Object
 */

class ntkData{
 protected:
  ntkIntDimension m_dataSize;
  ntkFloatDimension m_dataThickness;
  void* m_buffer;
  int m_bufferflag;
  ntkDataType m_dataType;
  int m_dimension;

  bool m_lockBuffer;
 public:
  ntkData(){};
  virtual ~ntkData(void){};

  virtual void readFile(const char* filename)=0;
  virtual void writeFile(const char* filename)=0;

  void* getBuffer(void){return m_buffer;}
  
  ntkDataType getDataType(){return m_dataType;}
  void setDataType(ntkDataType dataType){m_dataType=dataType;}
  int getDimension(){return m_dimension;}
  
  ntkIntDimension getDataSize(){return m_dataSize;}
  ntkFloatDimension getDataThickness(){return m_dataThickness;}
  void setDataThickness(ntkFloatDimension dataThickness){m_dataThickness=dataThickness;}
};

//include sub-classes

#include "ntk2DData.h"
#include "ntk2DData16.h"
#include "ntk3DData.h"
#include "ntk3DData16.h"
#include "ntk4DData.h"
#include "ntk4DData16.h"

#endif
