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

#ifdef BLISNN
#include "blis/blis.h"
#endif

Matrix::Matrix(int nRows, int nCols, int random = 0)
        : r(nRows), c(nCols) {

	mat = vector<float>(nRows * nCols);
	float scal_fac = 0.05; // randomize between (-1;1)
	if (random==0){
		float zero = 0.0;
		bli_ssetm( BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
				r, c, &zero, &mat[0], c, 1);
	} else if (random==1){
		bli_srandm(0, BLIS_DENSE, r, c, &mat[0], c, 1);
		bli_sscalm( BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
				r, c, &scal_fac, &mat[0], c, 1);
	}
}
		
Matrix Matrix::transpose() const {
    Matrix result(c, r, 0); // Initialize a new matrix with inverted dimension values

	// m = r, n = c
	// rs = 1, cs = m, rsf = 1, csf = n
    //printf("BLIS copy %dx%d\n",c,r);
    bli_scopym( 0, BLIS_NONUNIT_DIAG, BLIS_DENSE, BLIS_TRANSPOSE,
		c, r, const_cast<float*>(&mat[0]), c, 1, &result.mat[0], r, 1);  
    return result;
}

void Matrix::show() const {

	char e[5] = "";
	char format[8] = "%4.4f";
	bli_sprintm( e, r, c, const_cast<float*>(&mat[0]), c, 1, format, e );
}


void Matrix::mvp(const Vector &x, Vector &y) const {
	assert( c==x.size() ); 
	assert( r==y.size() );

	float alpha = 1.0;
	float beta = 0.0;
	//printf("BLIS gemv %dx%d\n",c,r);
	bli_sgemv( BLIS_NO_TRANSPOSE, BLIS_NO_CONJUGATE,
		   r, c, &alpha, const_cast<float*>(&mat[0]),
		   c, 1, const_cast<float*>( x.data() ) /* &t.vals[0] */,
		   // c, 1, &t.vals[0] ,
		   1, &beta, &y.vals[0], 1 );  

}

void Matrix::mvpt(const Vector &x, Vector &y) const {
	assert( r==x.size() );
	assert( c==y.size() );

	float alpha = 1.0;
	float beta = 0.0;
	//printf("BLIS gemv %dx%d\n",c,r);
	bli_sgemv( BLIS_TRANSPOSE, BLIS_NO_CONJUGATE,
		   r, c, &alpha, const_cast<float*>(&mat[0]), // test if r and c need to be flipped
		   //c, 1, &t.vals[0],
		   c, 1, const_cast<float*>( x.data() ),
		   1, &beta, &y.vals[0], 1 );  
}


void Matrix::outerProduct(const Vector &x, const Vector &y) {
	assert( x.size() == r );
	assert( y.size() == c );
	float val = 1.0;

	//printf("BLIS gemm (outer) %dx%d\n",r,c);
	bli_sgemm( BLIS_NO_TRANSPOSE, BLIS_TRANSPOSE,
			r, c, 1, &val, 
			const_cast<float*>(x.data()), 1, 1, 
			const_cast<float*>(y.data()), c, 1, &val, &mat[0], c, 1);

}


void Matrix::mmp(const Matrix &x, Matrix &y) const { // In place matrix matrix multiplication
	assert( c==x.r );
	assert( r==y.r );
	assert( x.c==y.c );

	float alpha = 1.0;
	float beta = 0.0;
	// m = r, n = x.c, k = c
	//printf("BLIS gemm %dx%dx%d\n",r,x.c,c);
	bli_sgemm( BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, 
		   r, x.c, c, &alpha, const_cast<float*>(&mat[0]),
		   //c, 1, &t.mat[0],
		   c, 1, const_cast<float*>( x.data() ),
		   x.c, 1, &beta, &y.mat[0], x.c, 1);
}

void Matrix::axpy( float a,const Matrix &x ) {
  assert( r==x.r );
  assert( c==x.c );
  const int n = nelements();
  assert( n==x.nelements() );

  bli_saxpyv( BLIS_NO_CONJUGATE,
	      n, &a, const_cast<float*>( x.data() ),1,
	      data(),1 );
};

float Matrix::normf() const {
  float norm;
  auto r = rowsize(), c = colsize();
  const auto mval = values().data();
  bli_snormfv( r*c, const_cast<float*>(mval), 1, &norm );
  return norm;
};
