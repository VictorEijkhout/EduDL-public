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
		
// VectorBatch VectorBatch::transpose() const {
//     const int c = batch_size(), r = item_size();
//     VectorBatch result(c, r, 0); // Initialize a new matrix with inverted dimension values

// 	bli_scopym( 0, BLIS_NONUNIT_DIAG, BLIS_DENSE, BLIS_TRANSPOSE,
// 		    c, r, const_cast<float*>(&vals[0]), c, 1, &result.vals[0], r, 1);  
//     return result;
// }

void VectorBatch::show() const {

    const int c = batch_size(), r = item_size();
	char e[5] = "";
	char forvals[8] = "%4.4f";
	bli_sprintm( e, r, c, const_cast<float*>(&vals[0]), c, 1, forvals, e );
}


/*
 * For explanation of the BLIS routines, see
 * https://github.com/flame/blis/blob/master/docs/BLISTypedAPI.md#gemm
 * and
 * https://github.com/flame/blis/blob/master/docs/BLISTypedAPI.md#computational-function-reference
 */
void VectorBatch::v2mp(const Matrix &m, VectorBatch &y) const {
  const int
    xr = item_size(),   xc = batch_size(),   // column storage
    mr = m.rowsize(),   mc = m.colsize(),    // row storage
    yr = y.item_size(), yc = y.batch_size(); // column

  if (trace_scalars())
    cout << "matrix vector product "
	 << mr << "x" << mc
	 << " & "
	 << xr << "x" << xc
	 << " => " << yr << "x" << yc
	 << endl;

  assert( xc==yc );
  assert( yr==mr );
  assert( mc==xr );

  const auto& mmat  = m.values().data();
  const auto& xvals = vals_vector().data();
  auto  yvals       = y.vals_vector().data();

  float alpha = 1.0;
  float beta = 0.0;
  bli_sgemm( BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, 
	     yr,yc,mc,
	     &alpha,
	     const_cast<float*>(mmat),  /* rsa,csa */ mr,1,
	     const_cast<float*>(xvals), /* rsb,csb */ 1,xr,
	     &beta,
	     yvals,                     /* rsc,csc */ 1,yr
	     );
}

void VectorBatch::v2tmp(const Matrix &x, VectorBatch &y) const {

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

// matrix transpose x self => y
void VectorBatch::v2mtp(const Matrix &m, VectorBatch &y) const {
  const int
    xr = item_size(),   xc = batch_size(),   // column storage
    mr = m.rowsize(),   mc = m.colsize(),    // row storage
    yr = y.item_size(), yc = y.batch_size(); // column

  if (trace_scalars())
    cout << "matrix transpose vector product "
	 << mr << "x" << mc
	 << " & "
	 << xr << "x" << xc
	 << " => " << yr << "x" << yc
	 << endl;

  assert( xc==yc );
  assert( yr==mc );
  assert( mr==xr );

  const auto& mmat  = m.values().data();
  const auto& xvals = vals_vector().data();
  auto        yvals = y.vals_vector().data();

  float alpha = 1.0;
  float beta = 0.0;
  bli_sgemm( BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, 
	     yr,yc,mr,
	     &alpha,
	     const_cast<float*>(mmat),  /* rsa,csa */ 1,mc,
	     const_cast<float*>(xvals), /* rsb,csb */ 1,xr,
	     &beta,
	     yvals,                     /* rsc,csc */ 1,yr
	     );

}

/*
 * x times self => m
 */
void VectorBatch::outer2(const VectorBatch &x, Matrix &m) const {
  const int
    yr = item_size(),   yc = batch_size(),   // column storage
    mr = m.rowsize(),   mc = m.colsize(),    // row storage
    xr = x.item_size(), xc = x.batch_size(); // column

  if (trace_scalars())
    cout << "outer product "
	 << xr << "x" << xc
	 << " & "
	 << yr << "x" << yc
	 << " => " << mr << "x" << mc
	 << endl;

  assert( yc==xc );
  assert( xr==mr );
  assert( yr==mc );
  const auto& xvals = x.vals_vector().data();
  const auto& yvals =   vals_vector().data();
  auto        mmat  = m.values().data();

  float alpha = 1.0;
  float beta = 0.0;
  bli_sgemm( BLIS_NO_TRANSPOSE, BLIS_TRANSPOSE, 
	     mr,mc,yc,
	     &alpha,
	     const_cast<float*>(xvals), /* rsa,csa */ 1,xr,
	     const_cast<float*>(yvals), /* rsb,csb */ 1,yr,
	     &beta,
	     mmat,                      /* rsc,csc */ mc,1
	     );

}
