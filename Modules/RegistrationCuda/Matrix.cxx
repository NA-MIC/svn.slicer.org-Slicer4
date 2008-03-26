#include "Matrix.h"

Matrix::Matrix()
{
    std::vector<double> col;
    col.push_back(0.0);
    mat.push_back(col);
}
Matrix::~Matrix()
{
}
Matrix::Matrix(int rows, int columns)
{
    std::vector<double> cols;
    cols.resize(columns, 0.0);
    mat.resize(rows, cols);
}
Matrix::Matrix(int rows, int columns, double value)
{
    std::vector<double> cols;
    cols.resize(columns, value);
    mat.resize(rows, cols);
}


Matrix Matrix::inverse() const
{
}

Matrix Matrix::pseudo_inverse() const
{
}

Matrix Matrix::operator*(const Matrix& other) const
{
    if (this->cols() != other.rows())
        return Matrix();
    Matrix mat(this->rows(), other.cols());

    for (int i = 0; i < mat.cols(); i++)
    {
        for (int j = 0; j < mat.rows(); j++)
        {
            for (int r = 1; r < this->cols(); r++)
            {
                mat(i,j) = (*this)(i, r) * other(r, j);
            }
        }
    }
    return mat;
}
