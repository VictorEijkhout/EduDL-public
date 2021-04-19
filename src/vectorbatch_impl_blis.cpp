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

#ifdef BLISNN
#include "blis/blis.h"
#endif

VectorBatch::VectorBatch(int nRows, int nCols, bool random)
  : r(nRows), c(nCols) {

  vals = vector<float>(nRows * nCols);
  float scal_fac = 0.05; // randomize between (-scal;scal)
  if (not random){
    float zero = 0.0;
    bli_ssetm( BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
	       r, c, &zero, &vals[0], c, 1);
  } else if (random){
    bli_srandm(0, BLIS_DENSE, r, c, &vals[0], c, 1);
    bli_sscalm( BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
		r, c, &scal_fac, &vals[0], c, 1);
  }
}
		
VectorBatch VectorBatch::transpose() const {
  VectorBatch result(c, r, 0); // Initialize a new matrix with inverted dimension values

  bli_scopym( 0, BLIS_NONUNIT_DIAG, BLIS_DENSE, BLIS_TRANSPOSE,
	      c, r, const_cast<float*>(&vals[0]), c, 1, &result.vals[0], r, 1);  
  return result;
}

void VectorBatch::show() const {

  char e[5] = "";
  char forvals[8] = "%4.4f";
  bli_sprintm( e, r, c, const_cast<float*>(&vals[0]), c, 1, forvals, e );
}


void VectorBatch::v2mp(const Matrix &x, VectorBatch &y) const { // In place matrix matrix multiplication
  // (m,n) * (n,k) -> (m,k)
  assert( c==x.r );
  assert( r==y.r );
  assert( x.c==y.c );

  float alpha = 1.0;
  float beta = 0.0;
  //printf("BLIS gemm %dx%dx%d\n",r,x.c,c);
  bli_sgemm( BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, 
	     r, x.c, c, &alpha, const_cast<float*>(&vals[0]),
	     c, 1, const_cast<float*>( x.data() ),
	     x.c, 1, &beta, &y.vals[0], x.c, 1);
}

void VectorBatch::v2tmp(const Matrix &x, VectorBatch &y) const { // In place matrix matrix multiplication
  // (n,m) * (n,k) -> (m,k)
  assert( r==x.r );
  assert( c==y.r );
  assert( x.c==y.c );

  float alpha = 1.0;
  float beta = 0.0;
  //printf("BLIS gemm %dx%dx%d\n",r,x.c,c);
  bli_sgemm( BLIS_TRANSPOSE, BLIS_NO_TRANSPOSE, 
	     c, x.c, r, &alpha, const_cast<float*>(&vals[0]),
	     c, 1, const_cast<float*>( x.data() ),
	     x.c, 1, &beta, &y.vals[0], x.c, 1);
}

void VectorBatch::v2mtp(const Matrix &x, VectorBatch &y) const { // In place matrix matrix multiplication
  // (m,n) * (k,n) -> (m,k)
  assert( r==y.r );
  assert( c==x.c );
  assert( x.r==y.c );

  float alpha = 1.0;
  float beta = 0.0;
  //printf("BLIS gemm %dx%dx%d\n",r,x.c,c);
  bli_sgemm( BLIS_NO_TRANSPOSE, BLIS_TRANSPOSE, 
	     y.r, y.c, c, &alpha, const_cast<float*>(&vals[0]),
	     c, 1, const_cast<float*>( x.data() ),
	     x.c, 1, &beta, &y.vals[0], x.c, 1);
}

void VectorBatch::outer2(const VectorBatch &x, Matrix &y) const { // In place matrix matrix multiplication
  // (n,m) *(n,k) -> (m,k)
  assert( r == x.r );
  assert( c == y.r );
  assert( x.c == y.c );

  float alpha = 1.0;
  float beta = 0.0;
  //printf("BLIS gemm %dx%dx%d\n",r,x.c,c);
  bli_sgemm( BLIS_TRANSPOSE, BLIS_NO_TRANSPOSE, 
	     c, x.c, r, &alpha, const_cast<float*>(&vals[0]), 
	     c, 1, const_cast<float*>( x.data() ),
	     x.c, 1, &beta, &y.mat[0], x.c, 1);
}
