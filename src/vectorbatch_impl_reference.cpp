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


