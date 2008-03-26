#ifndef NTKMATRIX_H
#define NTKMATRIX_H

#include <stdio.h>
//#include <octave/config.h>
#include <Matrix.h>
#include "ntkProperties.h"
#include "ntk3DData.h"

class ntkMatrix{
 public:

  /**
   * Object constructor. Create tensor with x*y*z size.
   */

  ntkMatrix(int x, int y);

  /**
   * Object constructor. Create tensor from inputData. inputData can be and should be destructed from user program once tensor is created. Matrix buffer is a copy of inputData buffer. 
   */

  ntkMatrix(ntk2DData* inputData);

  /**
   * Object constructor. Create tensor from inputBuffer. inputBuffer can be and should be destructed from user program once tensor is created. Matrix buffer is a copy of inputBuffer. 
   */

  ntkMatrix(float *inputBuffer, ntkIntDimension inputSize);

  /**
   * Object destructor.
   */

  ~ntkMatrix();
  
  /**
   * Initialize tensor. Set all values to 0.
   */

  void initialize();

  /**
   * Multiple tensor with matrix from left direction (Result=M*T).
   */
  
  ntkMatrix* multiplyMatrixLeft(Matrix matA);

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

  void printMatrix();

  /**
   * Set the value of (x,y,z) to value.
   */

  void setValue(int x, int y, float value);

  /**
   * Get the value of (x,y,z).
   */

  float getValue(int x, int y);

  /**
   * Get the size of tensor.
   */

  ntkIntDimension getMatrixSize();

  /**
   * Get tensor pointer. Matrix pointer should not be released from user program.
   */

  float* getMatrixPointer();
 protected:
  ntkIntDimension m_matrixSize;
  float *m_matrix;
  float *m_matTemp;
};

#endif
