#include "vector2.h"
#include <iostream>
#include <vector>
#include <algorithm>

using std::vector;

#ifdef BLISNN
#include "blis/blis.h"
#endif

VectorBundle::VectorBundle(int nRows, int nCols, bool random)
  : r(nRows), c(nCols) {

  vals = vector<float>(nRows * nCols);
  float scal_fac = 0.05; // randomize between (-1;1)
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
		
VectorBundle VectorBundle::transpose() const {
  VectorBundle result(c, r, 0); // Initialize a new matrix with inverted dimension values

  bli_scopym( 0, BLIS_NONUNIT_DIAG, BLIS_DENSE, BLIS_TRANSPOSE,
	      c, r, const_cast<float*>(&vals[0]), c, 1, &result.vals[0], r, 1);  
  return result;
}

void VectorBundle::show() const {

  char e[5] = "";
  char forvals[8] = "%4.4f";
  bli_sprintm( e, r, c, const_cast<float*>(&vals[0]), c, 1, forvals, e );
}


