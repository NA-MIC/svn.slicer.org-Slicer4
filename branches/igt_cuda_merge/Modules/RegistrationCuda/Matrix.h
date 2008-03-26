#ifndef __Matrix_h__
#define __Matrix_h__

#include <vector>

class Matrix
{
public:
    Matrix();
    ~Matrix();
    Matrix(int rows, int columns);
    Matrix(int rows, int columns, double value);

    int rows() const;
    int cols() const;

    double& operator()(int row, int column);

};

#endif __Matrix_h__
