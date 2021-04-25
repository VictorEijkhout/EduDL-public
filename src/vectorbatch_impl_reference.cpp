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

#include "vector2.h"
#include <iostream>
#include <vector>
#include <algorithm>

using std::vector;

VectorBatch::VectorBatch(int nRows, int nCols, bool random)
  : r(nRows), c(nCols) {

  vals = vector<float>(nRows * nCols);
  int i, j;
  if (not random){
    std::fill(vals.begin(), vals.end(), 0);
  }else if (random){
    for (i=0; i<nRows * nCols;i++){
      vals[i] = -0.1 + static_cast <float> (rand()) /( static_cast <float>(RAND_MAX/(0.1-(-0.1))));
    }
  }
}
		
VectorBatch VectorBatch::transpose() const {
  VectorBatch result(c, r, 0); // Initialize a new matrix with inverted dimension values
  int i1, i2; // Old and new index
  for (int i = 0; i < r; i++) {
    for (int j = 0; j < c; j++) {
      i1 = i * c + j; // Old indexing
      i2 = j * r + i; // New indexing

      result.vals[i2] = vals[i1]; // Move transposed values to new array
    }
  }
  return result;
}

void VectorBatch::show() const {
  int i, j;
  for (i = 0; i < r; i++) {
    for (j = 0; j < c; j++) {
      std::cout << vals[i * c + j] << ' ';
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}


void VectorBatch::v2mp(const Matrix &x, VectorBatch &y) const {
  assert( c == x.r );
  assert( r == y.r );
  assert( x.c == y.c );
  float sum;
  for (int i = 0; i < r; i++) { // Matrix multiplication subroutine
    for (int j = 0; j < x.c; j++) {
      sum = 0.0;
      for (int k = 0; k < x.r; k++) {
	sum += vals[i * c + k] * x.mat[k * x.c + j];
      }
      y.vals[i * y.c + j] = sum;
    }
  }
}


void VectorBatch::v2mtp(const Matrix &x, VectorBatch &y) const { // In place matrix matrix multiplication
  // (m,n) * (k,n) -> (m,k)
  assert( r == y.r );
  assert( c == x.c );
  assert( x.r == y.c );
  float sum;
  for (int i = 0; i < r; i++) { // Matrix multiplication subroutine
    for (int j = 0; j < x.c; j++) {
      sum = 0.0;
      for (int k = 0; k < x.r; k++) {
	sum += vals[i * c + k] * x.mat[j * x.c + k];
      }
      y.vals[i * y.c + j] = sum;
    }
  }

}

void VectorBatch::outer2(const VectorBatch &x, Matrix &y ) const {
  // (n,m) *(n,k) -> (m,k)
  assert( r == x.r );
  assert( c == y.r );
  assert( x.c == y.c );
  float sum;
  for (int i = 0; i < r; i++) { // Matrix multiplication subroutine
    for (int j = 0; j < x.c; j++) {
      sum = 0.0;
      for (int k = 0; k < x.r; k++) {
	sum += vals[k * c + i] * x.vals[j * x.c + k];
      }
      y.mat[i * y.c + j] = sum;
    }
  }

}
