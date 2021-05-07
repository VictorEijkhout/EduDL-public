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

Matrix::Matrix(int nRows, int nCols, int random = 0)
  : r(nRows), c(nCols) {

  mat = vector<float>(nRows * nCols);
  int i, j;
  if (random==0) {
    std::fill(mat.begin(), mat.end(), 0);
  } else if (random==1) {
    //std::fill(mat.begin(), mat.end(), .5);
    for (i=0; i<nRows * nCols;i++){
      //mat[i] = -0.1 + static_cast <float> (rand()) /( static_cast <float>(RAND_MAX/(0.1-(-0.1))));
      mat[i] = -0.1 + static_cast <float> (rand()) /( static_cast <float>(RAND_MAX) );
    }
  }

}
		
Matrix Matrix::transpose() const {
    Matrix result(c, r, 0); // Initialize a new matrix with inverted dimension values
    int i1, i2; // Old and new index
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            i1 = i * c + j; // Old indexing
            i2 = j * r + i; // New indexing

            result.mat[i2] = mat[i1]; // Move transposed values to new array
        }
    }

    return result;
}

void Matrix::show() const {

    int i, j;
    for (i = 0; i < r; i++) {
        for (j = 0; j < c; j++) {
            std::cout << mat[i * c + j] << ' ';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

}


void Matrix::mvp(const Vector &x, Vector &y) const {
	assert( c==x.size() ); 
	assert( r==y.size() );
    for (int i = 0; i < r; i++) {
        float sum = 0.0;
        for (int j = 0; j < c; j++) {
            sum += x.vals[j] * mat[i * c + j];
        }
        y.vals[i] = sum;
    }

}

void Matrix::mvpt(const Vector &x, Vector &y) const {
	assert( r==x.size() );
	assert( c==y.size() );
    for (int i = 0; i < r; i++) {
        float sum = 0.0;
        for (int j = 0; j < c; j++) {
            sum += x.vals[j] * mat[i * c + j];
        }
        y.vals[i] = sum;
    }

}


void Matrix::outerProduct(const Vector &x, const Vector &y) {
	assert( x.size() == r );
	assert( y.size() == c );
	float val = 1.0;
	for (int i = 0; i < x.size(); i++) {
        for (int j = 0; j < y.size(); j++) {
            mat[i * c + j] = x.vals[i] * y.vals[j];
        }
    }

}

void Matrix::mmp(const Matrix &x, Matrix &y) const {
	assert( c==x.r );
	assert( r==y.r );
	assert( x.c==y.c );
    float sum;
    for (int i = 0; i < r; i++) { // Matrix multiplication subroutine
        for (int j = 0; j < x.c; j++) {
            sum = 0.0;
            for (int k = 0; k < x.r; k++) {
                sum += mat[i * c + k] * x.mat[k * x.c + j];
            }
            y.mat[i * y.c + j] = sum;
        }
    }
}

void Matrix::axpy( float a,const Matrix &x ) {
  assert( r==x.r );
  assert( c==x.c );
  const int n = nelements();
  assert( n==x.nelements() );

  float *ydata = this->data();
  const auto& xdata = x.data();
  for (int i=0; i<n; i++) {
    ydata[i] += a * xdata[i];
  }

};

float Matrix::normf() const {
  float norm{0.f};
  // auto r = rowsize(), c = colsize();
  // const auto mval = values().data();
  for ( auto e : values() )
    norm += e*e;
  return sqrt(norm);
};
