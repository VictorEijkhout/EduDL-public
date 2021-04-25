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

#ifndef SRC_FUNCS_H
#define SRC_FUNCS_H
#include "matrix.h"
#include "vector.h"
#include "vector2.h"

#ifdef USE_GSL
#include "gsl/gsl-lite.hpp"
#endif

enum acFunc{RELU,SIG,SMAX,NONE};

//template <typename VectorBatch>
void relu_io    (const VectorBatch &i, VectorBatch &v);
//template <typename VectorBatch>
void sigmoid_io (const VectorBatch &i, VectorBatch &v);
//template <typename VectorBatch>
void softmax_io (const VectorBatch &i, VectorBatch &v);
//template <typename VectorBatch>
void linear_io    (const VectorBatch &i, VectorBatch &v);

//template <typename VectorBatch>
void reluGrad_io(const VectorBatch &m, VectorBatch &a);
//template <typename VectorBatch>
void sigGrad_io (const VectorBatch &m, VectorBatch &a);
//template <typename VectorBatch>
void smaxGrad_io(const VectorBatch &m, VectorBatch &a);
//template <typename VectorBatch>
void linGrad_io	(const VectorBatch &m, VectorBatch &a);

#ifdef USE_GSL
Matrix smaxGrad_vec( const gsl::span<float> &v);
#else
Matrix smaxGrad_vec( const std::vector<float> &v);
#endif

template <typename V>
static inline std::vector< std::function< void(const V&, V&) > > apply_activation{
  [] ( const V &v, V &a ) { relu_io(v,a); },
  [] ( const V &v, V &a ) { sigmoid_io(v,a); },
  [] ( const V &v, V &a ) { softmax_io(v,a); },
  [] ( const V &v, V &a ) { linear_io(v,a); }
};
  	
template <typename V>
static inline std::vector< std::function< void(const V&, V&) > > activate_gradient{
  [] (  const V &m, V &v ) { reluGrad_io(m,v); },
  [] (  const V &m, V &v ) { sigGrad_io(m,v); },
  [] (  const V &m, V &v ) { smaxGrad_io(m,v); },
  [] (  const V &m, V &v ) { linGrad_io(m,v); }
};

#endif //SRC_FUNCS_H
