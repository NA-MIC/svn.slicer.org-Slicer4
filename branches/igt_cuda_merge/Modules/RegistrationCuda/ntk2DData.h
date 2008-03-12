#ifndef NTK2DDATA_H
#define NTK2DDATA_H

#include "ntkData.h"

/**
 *2D Data Class Object
 */

class ntk2DData:public ntkData{
 public:
  ntk2DData(ntkIntDimension dataSize);
  ntk2DData(ntkIntDimension dataSize, ntkFloatDimension dataThickness);
  ntk2DData(ntkIntDimension dataSize, unsigned char* buffer);
  ntk2DData(ntkIntDimension dataSize, ntkFloatDimension dataThickness, unsigned char* buffer);
  ~ntk2DData(void);
  void readFile(const char* filename);
  void writeFile(const char* filename);
  void readBuffer(unsigned char* inBuffer);
  void writeBuffer(unsigned char* outBuffer);

  unsigned char* getBuffer(void);
  void setValue(ntkIntDimension dataPos, unsigned char inValue);
  unsigned char getValue(ntkIntDimension dataPos);
};

#endif
