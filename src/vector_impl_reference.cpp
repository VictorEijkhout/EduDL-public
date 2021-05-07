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

#include <algorithm>
#include "vector.h"
#include <iostream>

#include <cassert>

Vector::Vector(int s, int init) {
	r = s;
  vals = std::vector<float>(s);
  if (init==0){
    std::fill(vals.begin(),vals.end(), 0);
  }else if (init==1){
    for (int i=0; i<size(); i++){
      vals[i] = -0.1 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(0.1-(-0.1))));
    }
  }

}

void Vector::add( const Vector &v1 ) {
  assert(v1.size()==this->size());
  for (int i=0; i<size(); i++){
    vals[i] += v1.vals[i];
  }
}

void Vector::set_ax( float a, Vector &x) {
  assert(x.size()==this->size());
    for (int i=0; i<size(); i++){
        vals[i] = a * x.vals[i];
    }

}


void Vector::zeros() {
    std::fill(vals.begin(),vals.end(),0);

}

void Vector::show() {
	int i;
    for (i=0;i<size();i++) {
        std::cout << vals[i] << '\n';
    }
    std::cout << '\n';

}


