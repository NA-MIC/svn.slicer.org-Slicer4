#ifndef NTK2DDATA16_H
#define NTK2DDATA16_H

#include "ntkData.h"

/**
 *2D Data Class Object
 */

class ntk2DData16:public ntkData{
 public:
  ntk2DData16(ntkIntDimension dataSize);
  ntk2DData16(ntkIntDimension dataSize, ntkFloatDimension dataThickness);
  ntk2DData16(ntkIntDimension dataSize, unsigned short* buffer);
  ntk2DData16(ntkIntDimension dataSize, ntkFloatDimension dataThickness, unsigned short* buffer);
  ~ntk2DData16(void);
  void readFile(const char* filename);
  void writeFile(const char* filename);
  void readBuffer(unsigned short* inBuffer);
  void writeBuffer(unsigned short* outBuffer);
  unsigned short* getBuffer(void);
  void setValue(ntkIntDimension dataPos, unsigned short inValue);
  unsigned short getValue(ntkIntDimension dataPos);
};

#endif
