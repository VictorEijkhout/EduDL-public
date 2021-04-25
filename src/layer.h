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

#ifndef SRC_LAYER_H
#define SRC_LAYER_H

#include <functional>

#include "vector.h"
//#include "matrix.h"
#include "funcs.h"
#include "vector2.h"

class Net; // forward definition for friending
class Layer {
  friend class Net;
public:
  Layer();
  Layer(int insize, int outsize);

private: // but note that Net is a `friend' class!
  Vector biases; // Biases which come before the layer
  acFunc activation; // Activation functions of the layer
  Vector biased_product; // Values in the layer n after multiplying vals from n-1 and weights
  Matrix weights; // Weights which come before the layer
  Vector activated;
  VectorBatch activated_batch,delta,wdelta,dl;
  Vector d_activated; // for backpropagation
  VectorBatch biased_productm;
  VectorBatch d_activated_batch;
  
  // Vector dl; // dloss
  Matrix dW; // dw calculated per layer
  Matrix dw;		// cumulative dw
  Matrix dw_velocity; // For SGD with Momentum, RMSprop
  Vector db_velocity;
  Vector db;		// cumulative deltas
	
  Vector delta_mean; // mean of the deltas used in batch training
public:

  int input_size() const { return weights.r; };
  int output_size() const { return weights.c; };
  void set_initial_deltas( const Matrix&, const Vector& );
  void set_recursive_deltas( Vector &, const Layer&,const Layer& );
  //virtual void forward( const Vector &prevVals); // Virtual to support other types of layers later
  //virtual void backward(Vector &delta, const Matrix &W, const Vector &prev);
  void backward_update( const VectorBatch&, const VectorBatch& ,bool=false );
   
  void update_dw(VectorBatch &delta, VectorBatch prevValues);

  virtual void forward( const VectorBatch &prevVals);
  virtual void backward(VectorBatch &delta, const Matrix &W, const VectorBatch &prev);
		 
private:
  std::function< void(const VectorBatch&,VectorBatch&) > apply_activation_batch{
    [] ( const VectorBatch &v, VectorBatch &a ) { relu_io(v,a); } };
  std::function< void(const VectorBatch&,VectorBatch&) > activate_gradient_batch{
    [] ( const VectorBatch &v, VectorBatch &a ) { reluGrad_io(v,a); } };
public:
  void set_activation(acFunc f) {
    activation = f;
    apply_activation_batch  = apply_activation<VectorBatch>.at(f);
    activate_gradient_batch = activate_gradient<VectorBatch>.at(f);
  };
  void set_activation
  ( std::function< void(const VectorBatch&,VectorBatch&) > apply,
    std::function< void(const VectorBatch&,VectorBatch&) > activate ) {
    activation = acFunc::RELU;
    apply_activation_batch  = apply;
    activate_gradient_batch = activate;
  };
};


#endif //SRC_LAYER_H
