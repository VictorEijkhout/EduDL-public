//
// Created by Ilknur on 30-Dec-20.
//
#include <algorithm>
#include "vector.h"
#include <iostream>

#include <cassert>

#ifdef BLISNN
#include "blis/blis.h"
#endif

Vector::Vector(int s, int init) {
  r = s;

  vals = std::vector<float>(s);
  if (init==0){
    float zero = 0.0;
    bli_ssetv( BLIS_NO_CONJUGATE, s, &zero, &vals[0], 1);
  }else if (init==1)
    bli_srandv(s, &vals[0], 1);
}

void Vector::add(Vector &v1) {
  assert(v1.size()==this->size());

  bli_saddv( BLIS_NO_CONJUGATE, size(), &v1.vals[0], 1, &vals[0], 1 );
}

void Vector::set_ax( float a, Vector &x) {
  assert(x.size()==this->size());

  float b = static_cast<float>(a);
  bli_sscal2v(BLIS_NO_CONJUGATE, this->size(),&b,&x.vals[0],1,&(this->vals)[0], 1);
}


void Vector::zeros() {

  float zero = 0.0;
  bli_ssetv( BLIS_NO_CONJUGATE, size(), &zero, &vals[0], 1 ); // Set all values to 0
}

void Vector::show() {

  char sp[8] = " ";
  char format[8] = "%4.4f";
  bli_sprintm( sp, size(), 1, &vals[0], 1, size(), format, sp );
}


