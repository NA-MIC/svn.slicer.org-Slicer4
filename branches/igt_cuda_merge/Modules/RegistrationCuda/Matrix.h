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

    int rows() const { return mat.size(); }
    int cols() const { return (mat.empty()) ? 0 : mat[0].size() ; }

    Matrix inverse() const;
    Matrix pseudo_inverse() const;

    double& operator()(int row, int column) { return mat[row][column]; }
    const double& operator()(int row, int column) const { return mat[row][column]; }
    Matrix operator*(const Matrix& other) const;

private:
    std::vector<std::vector<double> > mat;
};

#endif __Matrix_h__
