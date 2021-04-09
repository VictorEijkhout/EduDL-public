//
// Created by Ilknur on 17-Dec-20.
//

#ifndef SRC_FUNCS_H
#define SRC_FUNCS_H
#include "matrix.h"
#include "vector.h"

enum acFunc{RELU,SIG,SMAX,NONE};

template <typename V>
void relu_io    (const V &i, V &v);
template <typename V>
void sigmoid_io (const V &i, V &v);
template <typename V>
void softmax_io (const V &i, V &v);
template <typename V>
void none_io    (const V &i, V &v);

template <typename V>
void reluGrad_io(const V &m, V &a);
template <typename V>
void sigGrad_io (const V &m, V &a);

#endif //SRC_FUNCS_H
