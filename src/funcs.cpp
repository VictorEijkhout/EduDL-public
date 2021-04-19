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

#include "funcs.h"

#include <vector>
using std::vector;

#include <cmath>
#include <cassert>

// V input output variants

template <typename V>
void relu_io(const V &m, V &a) {
  a.vals.assign(m.vals.begin(),m.vals.end());
  float alpha = 0.01; // used for leaky relu, for regular relu, simply set alpha to 0.0
  for (int i = 0; i < m.r * m.c; i++) {
    // values will be 0 if negative, and equal to themselves if positive
    if (a.vals[i] < 0)
      a.vals[i] *= alpha;
  }
}

template <typename V>
void sigmoid_io(const V &m, V &a) {
  a.vals.assign(m.vals.begin(),m.vals.end());
  for (int i = 0; i < m.r * m.c; i++) {
    // a.vals[i]*=(a.vals[i]>0); // values will be 0 if negative, and equal to themselves if positive
    a.vals[i] = 1 / (1 + exp(-a.vals[i]));
  }
}

template <typename V>
void softmax_io(const V &m, V &a) {

  vector<float> nB(m.r);
  vector<float> mV(m.r);

  for (int i = 0; i < m.r; i++) {
    nB.at(i) = 0.0;
    mV.at(i) = -9999;
  }

  for (int i = 0; i < m.r; i++) {
    for (int j = 0; j < m.c; j++) {
      if (m.vals.at(i + m.r * j) > mV.at(i)) {
	mV.at(i) = m.vals.at(i + m.r * j);
      }
    }
  }

  for (int i = 0; i < m.r; i++) {
    for (int j = 0; j < m.c; j++) {
      a.vals.at(i + m.r * j) = m.vals.at(i + m.r * j) - mV.at(i);
      a.vals.at(i + m.r * j) = exp(a.vals.at(i + m.r * j));
      nB.at(i) += a.vals.at(i + m.r * j);
    }
  }

  for (int i = 0; i < m.r; i++) {
    for (int j = 0; j < m.c; j++) {
      a.vals.at(i + m.r * j) = a.vals.at(i + m.r * j) / nB.at(i);
    }
  }

  for (int j=0; j < m.vals.size(); j++) {
    if (a.vals.at(j) <= 1e-7)
      a.vals.at(j) = 1e-7;
    if (a.vals.at(j) >= 1 - 1e-7)
      a.vals.at(j) = 1 - 1e-7;
  }

}

template <typename V>
void linear_io(const V &m, V &a) {
  a.vals.assign(m.vals.begin(),m.vals.end());
}

template <typename V>
void reluGrad_io(const V &m, V &a) {
  a.vals.assign(m.vals.begin(),m.vals.end());
  float alpha = 0.01;
  for ( auto &e : a.vals) {
    if (e<=0)
      e = alpha;
    else
      e = 1.0;
  }
}

template <typename V>
void sigGrad_io(const V &m, V &a) {
  assert( m.size()==a.size() );
  a.vals.assign(m.vals.begin(),m.vals.end());
  for ( auto &e : a.vals )
    e = e * (1.0 - e);
}

template <typename V>
void smaxGrad_io(const V &m, V &a) {
  assert( m.size()==a.size() );
  /* Incomplete for now */
}

Matrix smaxGrad_vec(std::vector<float> &v) {
  Matrix im(v.size(),1,0); // Input but converted to a matrix

  for (int i=0; i<v.size(); i++){
    im.mat[i] = v[i];
  }

  Matrix dM = im;

  Matrix diag(dM.r,dM.r,0);

  for (int i=0,j=0; i<diag.r*diag.c; i+=diag.r+1,j++) {
    // identity * dM
    diag.mat[i] = dM.mat[j];
  }

  // S_(i,j) dot S_(i,k)
  Matrix dMT = dM.transpose();

  Matrix S(dM.r,dMT.c,0);
  dM.mmp(dMT, S);
  im = diag - S; // Jacobian
  return im;

}


template <typename V>
void linGrad_io(const V &m, V &a) {
  assert( m.size()==a.size() );
  std::fill(a.vals.begin(), a.vals.end(), 1.0); // gradient of a linear function
}

// IM: Predefine templates so we can use them in separate .h and .cpp files
template void relu_io<Vector>(const Vector&, Vector&);
template void relu_io<VectorBatch>(const VectorBatch&, VectorBatch&);

template void sigmoid_io<Vector>(const Vector&, Vector&);
template void sigmoid_io<VectorBatch>(const VectorBatch&, VectorBatch&);

template void softmax_io<Vector>(const Vector&, Vector&);
template void softmax_io<VectorBatch>(const VectorBatch&, VectorBatch&);

template void linear_io<Vector>(const Vector&, Vector&);
template void linear_io<VectorBatch>(const VectorBatch&, VectorBatch&);

template void reluGrad_io<Vector>(const Vector&, Vector&);
template void reluGrad_io<VectorBatch>(const VectorBatch&, VectorBatch&);

template void sigGrad_io<Vector>(const Vector&, Vector&);
template void sigGrad_io<VectorBatch>(const VectorBatch&, VectorBatch&);

template void smaxGrad_io<Vector>(const Vector&, Vector&);
template void smaxGrad_io<VectorBatch>(const VectorBatch&, VectorBatch&);

template void linGrad_io<Vector>(const Vector&, Vector&);
template void linGrad_io<VectorBatch>(const VectorBatch&, VectorBatch&);
