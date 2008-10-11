#ifndef NTK3DDATA_H
#define NTK3DDATA_H

#include "ntkData.h"

/**
 *3D Data Class Object
 */

class ntk3DData : public ntkData{
 public:
  ntk3DData(ntkIntDimension dataSize);
  ntk3DData(ntkIntDimension dataSize, unsigned char initVal);
  ntk3DData(ntkIntDimension dataSize, ntkFloatDimension dataThickness);
  ntk3DData(ntkIntDimension dataSize, unsigned char* buffer);
  ntk3DData(ntkIntDimension dataSize, ntkFloatDimension dataThickness, unsigned char* buffer);
  ~ntk3DData(void);
  void readFile(const char* filename);
  void writeFile(const char* filename);
  void readNTKFile(const char* filename);
  void writeNTKFile(const char* filename);
  void readBuffer(unsigned char* inBuffer);
  void writeBuffer(unsigned char* outBuffer);
  unsigned char* getBuffer(void);
  void setValue(ntkIntDimension dataPos, unsigned char inValue);
  unsigned char getValue(ntkIntDimension dataPos);
  void setDataContrast(double center, double window);
  ntk3DData* getDataWithNewContrast(double center, double window);
  void flipXAxis(void);

  void lockBuffer(bool lock);
};

#endif
