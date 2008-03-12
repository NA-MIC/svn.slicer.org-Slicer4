#ifndef NTK4DDATA16_H
#define NTK4DDATA16_H

#include "ntkData.h"

/**
 *4D Data Class Object
 */

class ntk4DData16:public ntkData{
 public:
  ntk4DData16(ntkIntDimension dataSize);
  ntk4DData16(ntkIntDimension dataSize, ntkFloatDimension dataThickness);
  ntk4DData16(ntkIntDimension dataSize, unsigned short* buffer);
  ntk4DData16(ntkIntDimension dataSize, ntkFloatDimension dataThickness, unsigned short* buffer);
  ~ntk4DData16(void);

  void readFile(const char* filename);
  void writeFile(const char* filename);
  void readBuffer(unsigned short* inBuffer);
  void writeBuffer(unsigned short* outBuffer);
  unsigned short* getBuffer(void);
  void setValue(ntkIntDimension dataPos, unsigned short inValue);
  unsigned short getValue(ntkIntDimension dataPos);
};

#endif
