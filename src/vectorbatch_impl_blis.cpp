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
using std::cout;
using std::endl;
#include <vector>
#include <algorithm>

using std::vector;

#ifdef BLISNN
#include "blis/blis.h"
#endif

VectorBatch::VectorBatch(int nRows, int nCols, bool random) {
  allocate(nRows,nCols);
  const int r=nRows, c=nCols;

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
  const int c = batch_size(), r = item_size();
  VectorBatch result(c, r, 0); // Initialize a new matrix with inverted dimension values

  bli_scopym( 0, BLIS_NONUNIT_DIAG, BLIS_DENSE, BLIS_TRANSPOSE,
	      c, r, const_cast<float*>(&vals[0]), c, 1, &result.vals[0], r, 1);  
  return result;
}

void VectorBatch::show() const {

  const int c = batch_size(), r = item_size();
  char e[5] = "";
  char forvals[8] = "%4.4f";
  bli_sprintm( e, r, c, const_cast<float*>(&vals[0]), c, 1, forvals, e );
}


void VectorBatch::v2mp(const Matrix &x, VectorBatch &y) const { // In place matrix matrix multiplication
  // (m,n) * (n,k) -> (m,k)
  const int c = batch_size(), r = item_size();
  cout << "multiply matrix " << x.rowsize() << "x" << x.colsize()
       << " to vector " << item_size() << "x" << batch_size()
       << " into vector " << y.item_size() << "x" << y.batch_size()
       << endl;
  assert( x.colsize()==item_size() );
  assert( x.rowsize()==y.item_size() );
  assert( batch_size()==y.batch_size() );

  float alpha = 1.0;
  float beta = 0.0;
  //printf("BLIS gemm %dx%dx%d\n",r,x.item_size(),c);
  // y^t = x^t self^t => y = self x
  bli_sgemm( BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, 
	     y.batch_size(), y.item_size(), item_size(),
	     &alpha,
	     const_cast<float*>(data()),item_size(),1,
	     const_cast<float*>( x.data() ),x.colsize(),1,
	     &beta,
	     y.data(),y.item_size(),1
	     );
  // bli_sgemm( BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, 
  // 	   r, x.colsize(), c, &alpha, const_cast<float*>(&vals[0]),
  // 	   c, 1, const_cast<float*>( x.data() ),
  // 	   x.colsize(), 1, &beta, &y.vals[0], x.colsize(), 1);
}

void VectorBatch::v2tmp(const Matrix &x, VectorBatch &y) const { // In place matrix matrix multiplication
  // (n,m) * (n,k) -> (m,k)
  const int c = batch_size(), r = item_size();
  assert( r==x.rowsize() );
  assert( c==y.batch_size() );
  assert( x.colsize()==y.item_size() );

  float alpha = 1.0;
  float beta = 0.0;
  //printf("BLIS gemm %dx%dx%d\n",r,x.colsize(),c);
  bli_sgemm( BLIS_TRANSPOSE, BLIS_NO_TRANSPOSE, 
	     c, x.colsize(), r, &alpha, const_cast<float*>(&vals[0]),
	     c, 1, const_cast<float*>( x.data() ),
	     x.colsize(), 1, &beta, &y.vals[0], x.colsize(), 1);
}

void VectorBatch::v2mtp(const Matrix &x, VectorBatch &y) const { // In place matrix matrix multiplication
  // (m,n) * (k,n) -> (m,k)
  const int c = batch_size(), r = item_size();
  assert( r==y.batch_size() );
  assert( c==x.colsize() );
  assert( x.rowsize()==y.item_size() );

  float alpha = 1.0;
  float beta = 0.0;
  //printf("BLIS gemm %dx%dx%d\n",r,x.colsize(),c);
  bli_sgemm( BLIS_NO_TRANSPOSE, BLIS_TRANSPOSE, 
	     y.batch_size(), y.item_size(), c, &alpha, const_cast<float*>(&vals[0]),
	     c, 1, const_cast<float*>( x.data() ),
	     x.colsize(), 1, &beta, &y.vals[0], y.item_size(), 1);
}

void VectorBatch::outer2(const VectorBatch &x, Matrix &y) const { // In place matrix matrix multiplication
  // (n,m) *(n,k) -> (m,k)
  const int c = batch_size(), r = item_size();
  assert( r == x.batch_size() );
  assert( c == y.rowsize() );
  assert( x.item_size() == y.colsize() );

  float alpha = 1.0;
  float beta = 0.0;
  //printf("BLIS gemm %dx%dx%d\n",r,x.item_size(),c);
  bli_sgemm( BLIS_TRANSPOSE, BLIS_NO_TRANSPOSE, 
	     c, x.item_size(), r, &alpha, const_cast<float*>(&vals[0]), 
	     c, 1, const_cast<float*>( x.data() ),
	     x.item_size(), 1, &beta, y.data(), //&y.mat[0],
	     x.item_size(), 1);
}
