/****************************************************************
 ****************************************************************
 ****
 **** This text file is part of the source of 
 **** `Introduction to High-Performance Scientific Computing'
 **** by Victor Eijkhout, copyright 2012-2021
 ****
 **** Deep Learning Network code 
 **** copyright 2021 Ilknur Mustafazade
 ****
 ****************************************************************
 ****************************************************************/

#include "matrix.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>

using std::vector;

Matrix::Matrix() { // Default constructor
    r = 0;
    c = 0;
    mat.clear();
}

// void Matrix::flatten() { // Matrices are initialized in a coalesced flat 1D representation, so changing the dimension values are enough
//     r = r * c;
//     c = 1;
// }


float* Matrix::data() { return mat.data(); };
const float* Matrix::data() const { return mat.data(); };

Matrix &Matrix::operator=(const Matrix &m2) { // Overloading the = operator
    r = m2.r;
    c = m2.c;

    this->mat = m2.mat; // IM Since we're using vectors we can just use the assignment from that

    return *this;
}

Matrix Matrix::operator+(const Matrix &m2) const {
    Matrix out(m2.r, m2.c, 0);
    for (int i = 0; i < m2.r * m2.c; i++) {
        out.mat[i] = this->mat[i] + m2.mat[i];
    }
    return out;
}

Matrix Matrix::operator-(const Matrix &m2) {
    Matrix out(m2.r, m2.c, 0);
    for (int i = 0; i < m2.r * m2.c; i++) {
        out.mat[i] = this->mat[i] - m2.mat[i];
    }
    return out;
}



Matrix operator*(const float &c, const Matrix &m) {
    Matrix o = m;
    assert(o.mat.size()==o.r*o.c);
    for (int i = 0; i < o.r * o.c; i++) {
        o.mat[i] = c * o.mat[i];
    }
    return o;
}

Matrix operator/(const Matrix &m, const float &c) {
    Matrix o = m;
    for (int i = 0; i < o.r * o.c; i++) {
        o.mat[i] = o.mat[i] / c;
    }
    return o;
}


Matrix Matrix::operator*(const Matrix &m2) { // Hadamard product
    Matrix out(m2.r, m2.c, 0);
    assert(out.mat.size()==m2.r * m2.c);
    for (int i = 0; i < m2.r * m2.c; i++) {
        out.mat[i] = this->mat[i] * m2.mat[i];
    }
    return out;
}

Matrix Matrix::operator/(const Matrix &m2) { // Hadamard product
    Matrix out(m2.r, m2.c, 0);
    for (int i = 0; i < m2.r * m2.c; i++) {
        out.mat[i] = this->mat[i] / m2.mat[i];
    }
    return out;
}

Matrix Matrix::operator-() {
    Matrix result = *this;
    for (int i = 0; i < r * c; i++) {
        result.mat[i] = -mat[i];
    }

    return result;
};

void Matrix::addvh(const Vector &y) {
    for (int j = 0; j < c; j++) {
        for (int i = 0; i < y.size(); i++) {
            mat[j + c * i] += y.vals[i];
        }
    }
}

Vector Matrix::meanv() { // Returns a vector of column-wise means
    Vector mean(r, 0);
    float avg;
    for (int i = 0; i < r; i++) {
        avg = 0.0;
        for (int j = 0; j < c; j++) {
            avg += mat[i * c + j];
        }
        mean.vals[i] = avg;
    }
    return mean;
}


void Matrix::zeros() {
    std::fill(mat.begin(), mat.end(), 0);
}

void Matrix::square() {
    std::for_each(mat.begin(), mat.end(), [](auto &n) { n *= n;});
}
