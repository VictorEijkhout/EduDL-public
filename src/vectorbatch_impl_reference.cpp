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

#include "trace.h"
#include "vector2.h"
#include <iostream>
using std::cout;
using std::endl;
#include <vector>
using std::vector;
#include <algorithm>

#ifdef DEBUG
#define ELEMENTc(X,I,J,M,N) X.at( INDEXc(I,J,M,N) )
#define ELEMENTr(X,I,J,M,N) X.at( INDEXr(I,J,M,N) )
#else
#define ELEMENTc(X,I,J,M,N) X[ INDEXc(I,J,M,N) ]
#define ELEMENTr(X,I,J,M,N) X[ INDEXr(I,J,M,N) ]
#endif

VectorBatch::VectorBatch(int batchsize, int itemsize, bool random) {
  allocate(batchsize,itemsize);

  int i, j;
  if (not random){
    std::fill(vals.begin(), vals.end(), 0);
  }else if (random){
    for (i=0; i<batchsize * itemsize;i++){
      vals[i] = -0.1 + static_cast <float> (rand()) /( static_cast <float>(RAND_MAX/(0.1-(-0.1))));
    }
  }
}
		
// VectorBatch VectorBatch::transpose() const {
//     const int c = batch_size(), r = item_size();
//      // Initialize a new matrix with inverted dimension values
//     VectorBatch result(item_size(),batch_size(), 0);
//     int i1, i2; // Old and new index
//     for (int i = 0; i < r; i++) {
//         for (int j = 0; j < c; j++) {
//             i1 = i * c + j; // Old indexing
//             i2 = j * r + i; // New indexing

//             result.vals[i2] = vals[i1]; // Move transposed values to new array
//         }
//     }
//     return result;
// }

void VectorBatch::show() const {
    const int c = batch_size(), r = item_size();
    int i, j;
    for (i = 0; i < r; i++) {
        for (j = 0; j < c; j++) {
            std::cout << vals[i * c + j] << ' ';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}


// y = x x
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

  const auto& mmat = m.values();
  const auto& xvals = vals_vector();
  auto& yvals = y.vals_vector();
  int xi{0},yi{0},mi{0};
  for (int i = 0; i < yr; i++) { // Matrix multiplication subroutine
    for (int j = 0; j < yc; j++) {
      float sum = 0.0;
      for (int k = 0; k < mc; k++) {
	sum += ELEMENTr( mmat,i,k,mr,mc ) * ELEMENTc( xvals,k,j,xr,xc ) ;
	mi = INDEXr( i,k,mr,mc ); xi = INDEXc( k,j,xr,xc ) ;
	//sum += mmat.at( INDEXr(i,k,mr,mc) ) * xvals.at( INDEXc(k,j,xr,xc) ) ;
      }
      //yvals.at( INDEXc(i,j,yr,yc) ) = sum;
      ELEMENTc( yvals,i,j,yr,yc ) = sum;
      yi = INDEXc( i,j,yr,yc );
    }
  }
  assert( xi==xvals.size()-1 );
  assert( yi==yvals.size()-1 );
  assert( mi==mmat.size()-1 );
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

  const auto& mmat  = m.values();
  const auto& xvals = vals_vector();
  auto&       yvals = y.vals_vector();
    for (int i = 0; i < yr; i++) { // Matrix multiplication subroutine
        for (int j = 0; j < yc; j++) {
	    float sum = 0.0;
            for (int k = 0; k < mr; k++) {
	      //sum += mmat[ INDEXr(k,i,mc,mr) ] * xvals[ INDEXc(k,j,xr,xc) ]; 
	      // matrix is by rows, so transpose by columns!
	      sum += ELEMENTc( mmat,i,k,mc,mr ) * ELEMENTc( xvals,k,j,xr,xc ); 
            }
            yvals.at( INDEXc(i,j,yr,yc) )  = sum;
        }
    }

}

/*
 * x times self => m
 */
void VectorBatch::outer2(const VectorBatch &x, Matrix &m ) const {
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
  const auto& xvals = x.vals_vector();
  const auto& yvals =   vals_vector();
  auto& mmat = m.values();
    for (int i = 0; i < mr; i++) { // Matrix multiplication subroutine
      for (int j = 0; j < mc; j++) {
	    float sum = 0.0;
            for (int k = 0; k < yc; k++) {
	      /*
	       * index (k,j) in Y transpose => (j,k) in Y
	       */
	      //sum += yvals[ INDEXc(i,k,yr,yc) ] * xvals[ INDEXc(j,k,xr,xc) ];
	      sum += xvals.at( INDEXc(i,k,xr,xc) ) * yvals.at( INDEXc(j,k,yr,yc) );
            }
            //mmat[ INDEX(i,j,mr,mc) ] = sum;
            mmat.at( INDEXr(i,j,mr,mc) ) = sum;
        }
    }

}
