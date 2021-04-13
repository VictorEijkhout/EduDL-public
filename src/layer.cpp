//
// Created by Ilknur on 17-Dec-20.
//

#include "layer.h"
#include <iostream>

Layer::Layer() {};
Layer::Layer(int insize,int outsize)
  : weights( Matrix(outsize,insize,1) ),
    dw     ( Matrix(outsize,insize, 0) ),
    dW( Matrix(outsize,insize, 0) ),
    dw_velocity( Matrix(outsize,insize, 0) ),
    biases( Vector(outsize, 1 ) ),
    biased_product( Vector(outsize, 0) ),
    activated( Vector(outsize, 0) ),
    d_activated( Vector(outsize, 0) ),
    biased_productm( VectorBundle(outsize,insize,0) ),
    activated_batch( VectorBundle(outsize,insize, 0) ),
    am1a( VectorBundle(outsize,insize, 0) ),
    d_activated_batch ( VectorBundle(outsize,insize, 0) ),
    db( Vector(outsize, 0) ),
    delta_mean( Vector(outsize, 0) ),
    dl( Vector(outsize, 0) ),
    db_velocity( Vector(outsize, 0) ) {};



void Layer::forward(const Vector &prevVals) {
  weights.mvp(prevVals, biased_product); // Forward the data
  biased_product.add(biases);
	
  apply_activation<Vector>.at(activation)(biased_product, activated);

}

void Layer::forward(const VectorBundle &prevVals) {

  VectorBundle output(output_size(), prevVals.bundle_size(), 0);
  weights.mv2p(prevVals, output); // Forward the data

  output.addvh(biases); // Add the bias
	
  activated_batch = output;
  apply_activation<VectorBundle>.at(activation)(output, activated_batch);

}



void Layer::set_initial_deltas( const Matrix &dW, const Vector &delta ) {
  dw = dw + dW;
  db = db + delta;
};

void Layer::set_recursive_deltas( Vector &delta, const Layer &next,const Layer &prev ) {
  backward(delta, next.weights, prev.activated);
};

void Layer::backward(Vector &delta, const Matrix &W, const Vector &prev) {

  W.mvpt(delta, dl); // Derivative of the weights with respect to the loss in dl

  activate_gradient<Vector>.at(activation)(activated, d_activated);

  delta = d_activated * dl; // Derivative of the current layer

  dW.outerProduct(delta, prev);

  // Summing deltas over the batch to average later
  dw = dw + dW;
  db = db + delta;

}

void Layer::backward(VectorBundle &delta, const Matrix &W, const VectorBundle &prev) {

  VectorBundle dl(W.c, delta.bundle_size(), 0);
  W.mv2pt(delta, dl); // Derivative of the weights with respect to the loss in dl
	
  activate_gradient<VectorBundle>.at(activation)(activated_batch, d_activated_batch); 
  delta = d_activated_batch * dl; // Derivative of the current layer

  update_dw(delta, prev);
}

void Layer::update_dw(VectorBundle &delta, VectorBundle prevValues) {
  dW.outer2(delta,prevValues);
  dw = dw + dW;
  delta_mean = delta.meanv();
  db = db + delta_mean;	
}

void Layer::backward_update( const VectorBundle &prev_wdelta, const VectorBundle& prev_output,bool input_layer ) {
  std::cout << "here" << std::endl;
  //  delta = a1a * wdelta;
  std::cout << activation << std::endl; 
  activate_gradient<VectorBundle>.at(activation)(activated_batch, am1a); 
  delta = am1a * prev_wdelta; // Derivative of the current layer

  dw = dw + dW;
  Vector mdelta  = delta.meanv();
  db = db + mdelta;

  dW.outer2(delta,prev_output);
   
  if (not input_layer) {
    wdelta = VectorBundle(input_size(), delta.bundle_size(), 0);
    weights.mv2pt(delta, wdelta);
  }
};
