#ifndef NTK3DDATA16_H
#define NTK3DDATA16_H

#include "ntkData.h"
#include <string.h>

/**
 *3D Data Class Object
 */

class ntk3DData16 : public ntkData{
 public:
  ntk3DData16(ntkIntDimension dataSize);
  ntk3DData16(ntkIntDimension dataSize, ntkFloatDimension dataThickness);
  ntk3DData16(ntkIntDimension dataSize, unsigned short* buffer);
  ntk3DData16(ntkIntDimension dataSize, ntkFloatDimension dataThickness, unsigned short* buffer);
  ~ntk3DData16(void);

  void readFile(const char* filename);
  void writeFile(const char* filename);
  void readBuffer(unsigned short* inBuffer);
  void writeBuffer(unsigned short* outBuffer);
  unsigned short* getBuffer(void);
  void setValue(ntkIntDimension dataPos, unsigned short inputValue);
  unsigned short getValue(ntkIntDimension dataPos);
};

#endif
