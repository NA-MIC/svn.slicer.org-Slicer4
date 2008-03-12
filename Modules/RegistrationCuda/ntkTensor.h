#ifndef NTKTENSOR_H
#define NTKTENSOR_H

#include <stdio.h>
//#include <octave/config.h>
//#include <octave/Matrix.h>
#include "ntkProperties.h"
#include "ntk3DData.h"

class ntkTensor{
 public:

  /**
   * Object constructor. Create tensor with x*y*z size.
   */

  ntkTensor(int x, int y, int z);

  /**
   * Object constructor. Create tensor from inputData. inputData can be and should be destructed from user program once tensor is created. Tensor buffer is a copy of inputData buffer. 
   */

  ntkTensor(ntk3DData* inputData);

  /**
   * Object constructor. Create tensor from inputBuffer. inputBuffer can be and should be destructed from user program once tensor is created. Tensor buffer is a copy of inputBuffer. 
   */

  ntkTensor(float *inputBuffer, ntkIntDimension inputSize);

  /**
   * Object destructor.
   */

  ~ntkTensor();
  
  /**
   * Initialize tensor. Set all values to 0.
   */

  void initialize();

  /**
   * Multiple tensor with matrix from left direction (Result=M*T).
   */
  
//  ntkTensor* multiplyMatrixLeft(Matrix matA);

  /**
   * Switch tensor axis in forward direction (x->y, y->z, z->x).
   */

  void switchAxisForward();

  /**
   * Switch tensor axis in forward direction (x->z, y->x, z->y).
   */

  void switchAxisBackward();

  /**
   * Print tensor values to stdout.
   */

  void printTensor();

  /**
   * Set the value of (x,y,z) to value.
   */

  void setValue(int x, int y, int z, float value);

  /**
   * Get the value of (x,y,z).
   */

  float getValue(int x, int y, int z);

  /**
   * Get the size of tensor.
   */

  ntkIntDimension getTensorSize();

  /**
   * Get tensor pointer. Tensor pointer should not be released from user program.
   */

  float* getTensorPointer();
 protected:
  ntkIntDimension m_tensorSize;
  float *m_tensor;
  float *m_tenTemp;
};

#endif
