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

#include <iostream>
using std::cout;
using std:: endl;
#include <vector>
using std::vector;

#include <cmath>
#include <cassert>

// VectorBatch input output variants

//template <typename VectorBatch>
void relu_io(const VectorBatch &mm, VectorBatch &a) {
  VectorBatch m(mm);
  assert( a.item_size()==m.item_size() );
  assert( a.batch_size()==m.batch_size() );
  auto& avals = a.vals_vector();
  avals.assign(m.vals_vector().begin(),m.vals_vector().end());
  const float alpha = 0.01; // used for leaky relu, for regular relu, set alpha to 0.0
  for (int i = 0; i < m.batch_size() * m.item_size(); i++) {
    // values will be 0 if negative, and equal to themselves if positive
    if (avals.at(i) < 0)
      avals.at(i) *= alpha;
    //cout << i << ":" << avals.at(i) << endl;
  }
}

//codesnippet netsigmoid
//template <typename VectorBatch>
void sigmoid_io(const VectorBatch &m, VectorBatch &a) {
  a.vals_vector().assign(m.vals_vector().begin(),m.vals_vector().end());
  for (int i = 0; i < m.batch_size() * m.item_size(); i++) {
    // a.vals_vector()[i]*=(a.vals_vector()[i]>0); // values will be 0 if negative, and equal to themselves if positive
    a.vals_vector()[i] = 1 / (1 + exp(-a.vals_vector()[i]));
  }
}
//codesnippet end

//template <typename VectorBatch>
void softmax_io(const VectorBatch &m, VectorBatch &a) {

  vector<float> nB(m.batch_size());
  vector<float> mVectorBatch(m.batch_size());

  for (int i = 0; i < m.batch_size(); i++) {
    nB.at(i) = 0.0;
    mVectorBatch.at(i) = -9999;
  }

  for (int i = 0; i < m.batch_size(); i++) {
    for (int j = 0; j < m.item_size(); j++) {
      if (m.vals_vector().at(i + m.batch_size() * j) > mVectorBatch.at(i)) {
	mVectorBatch.at(i) = m.vals_vector().at(i + m.batch_size() * j);
      }
    }
  }

  for (int i = 0; i < m.batch_size(); i++) {
    for (int j = 0; j < m.item_size(); j++) {
      a.vals_vector().at(i + m.batch_size() * j) = m.vals_vector().at(i + m.batch_size() * j) - mVectorBatch.at(i);
      a.vals_vector().at(i + m.batch_size() * j) = exp(a.vals_vector().at(i + m.batch_size() * j));
      nB.at(i) += a.vals_vector().at(i + m.batch_size() * j);
    }
  }

  for (int i = 0; i < m.batch_size(); i++) {
    for (int j = 0; j < m.item_size(); j++) {
      a.vals_vector().at(i + m.batch_size() * j) = a.vals_vector().at(i + m.batch_size() * j) / nB.at(i);
    }
  }

  for (int j=0; j < m.vals_vector().size(); j++) {
    if (a.vals_vector().at(j) <= 1e-7)
      a.vals_vector().at(j) = 1e-7;
    if (a.vals_vector().at(j) >= 1 - 1e-7)
      a.vals_vector().at(j) = 1 - 1e-7;
  }

}

//template <typename VectorBatch>
void linear_io(const VectorBatch &m, VectorBatch &a) {
  a.vals_vector().assign(m.vals_vector().begin(),m.vals_vector().end());
}

//template <typename VectorBatch>
void reluGrad_io(const VectorBatch &m, VectorBatch &a) {
  a.vals_vector().assign(m.vals_vector().begin(),m.vals_vector().end());
  float alpha = 0.01;
  for ( auto &e : a.vals_vector()) {
    if (e<=0)
      e = alpha;
    else
      e = 1.0;
  }
}

//template <typename VectorBatch>
void sigGrad_io(const VectorBatch &m, VectorBatch &a) {
  assert( m.size()==a.size() );
  a.vals_vector().assign(m.vals_vector().begin(),m.vals_vector().end());
  for ( auto &e : a.vals_vector() )
    e = e * (1.0 - e);
}

//template <typename VectorBatch>
void smaxGrad_io(const VectorBatch &m, VectorBatch &a) {
  assert( m.size()==a.size() );
  /* Incomplete for now */
}

#ifdef USE_GSL
Matrix smaxGrad_vec( const gsl::span<float> &v)
#else
  Matrix smaxGrad_vec( const std::vector<float> &v)
#endif
{
  Matrix im(v.size(),1,0); // Input but converted to a matrix

  for (int i=0; i<v.size(); i++){
    float *i_data = im.data();
    *( i_data +i ) // im.mat[i]
      = v[i];
  }

  Matrix dM = im;

  Matrix diag(dM.rowsize(),dM.rowsize(),0);

  for (int i=0,j=0; i<diag.rowsize()*diag.colsize(); i+=diag.rowsize()+1,j++) {
    // identity * dM
    float *d_data = diag.data(), *m_data = dM.data();
    *( d_data+i ) // diag.mat[i]
      = *( m_data+j ); //dM.mat[j];
  }

  // S_(i,j) dot S_(i,k)
  Matrix dMT = dM.transpose();

  Matrix S(dM.rowsize(),dMT.colsize(),0);
  dM.mmp(dMT, S);
  im = diag - S; // Jacobian
  return im;

}


//template <typename VectorBatch>
void linGrad_io(const VectorBatch &m, VectorBatch &a) {
  assert( m.size()==a.size() );
  std::fill(a.vals_vector().begin(), a.vals_vector().end(), 1.0); // gradient of a linear function
}

// IM: Predefine templates so we can use them in separate .h and .item_size()pp files
/*template void relu_io<VectorBatchector>(const VectorBatchector&, VectorBatchector&);
  template void relu_io<VectorBatchectorBatch>(const VectorBatchectorBatch&, VectorBatchectorBatch&);

  template void sigmoid_io<VectorBatchector>(const VectorBatchector&, VectorBatchector&);
  template void sigmoid_io<VectorBatchectorBatch>(const VectorBatchectorBatch&, VectorBatchectorBatch&);

  template void softmax_io<VectorBatchector>(const VectorBatchector&, VectorBatchector&);
  template void softmax_io<VectorBatchectorBatch>(const VectorBatchectorBatch&, VectorBatchectorBatch&);

  template void linear_io<VectorBatchector>(const VectorBatchector&, VectorBatchector&);
  template void linear_io<VectorBatchectorBatch>(const VectorBatchectorBatch&, VectorBatchectorBatch&);

  template void reluGrad_io<VectorBatchector>(const VectorBatchector&, VectorBatchector&);
  template void reluGrad_io<VectorBatchectorBatch>(const VectorBatchectorBatch&, VectorBatchectorBatch&);

  template void sigGrad_io<VectorBatchector>(const VectorBatchector&, VectorBatchector&);
  template void sigGrad_io<VectorBatchectorBatch>(const VectorBatchectorBatch&, VectorBatchectorBatch&);

  template void smaxGrad_io<VectorBatchector>(const VectorBatchector&, VectorBatchector&);
  template void smaxGrad_io<VectorBatchectorBatch>(const VectorBatchectorBatch&, VectorBatchectorBatch&);

  template void linGrad_io<VectorBatchector>(const VectorBatchector&, VectorBatchector&);
  template void linGrad_io<VectorBatchectorBatch>(const VectorBatchectorBatch&, VectorBatchectorBatch&);*/
