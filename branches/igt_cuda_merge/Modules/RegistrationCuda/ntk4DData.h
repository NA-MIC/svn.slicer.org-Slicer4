#ifndef NTK4DDATA_H
#define NTK4DDATA_H

#include "ntkData.h"

/**
 *4D Data Class Object
 */

class ntk4DData:public ntkData{
 public:
  ntk4DData(ntkIntDimension dataSize);
  ntk4DData(ntkIntDimension dataSize, ntkFloatDimension dataThickness);
  ntk4DData(ntkIntDimension dataSize, unsigned char* buffer);
  ntk4DData(ntkIntDimension dataSize, ntkFloatDimension dataThickness, unsigned char* buffer);
  ~ntk4DData(void);

  void readFile(const char* filename);
  void writeFile(const char* filename);
  void readBuffer(unsigned char* inBuffer);
  void writeBuffer(unsigned char* outBuffer);
  unsigned char* getBuffer(void);
  void setValue(ntkIntDimension dataPos, unsigned char inValue);
  unsigned char getValue(ntkIntDimension dataPos);
};

#endif
