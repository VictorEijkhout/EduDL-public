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
    activatedm( VectorBundle(outsize,insize, 0) ),
    d_activatedm ( VectorBundle(outsize,insize, 0) ),
    db( Vector(outsize, 0) ),
    dl( Vector(outsize, 0) ),
    db_velocity( Vector(outsize, 0) ) {};



void Layer::forward(const Vector &prevVals) {
  //    Vector output(weights.r, 0);
  weights.mvp(prevVals, biased_product); // Forward the data
  biased_product.add(biases);
	
  apply_activation<Vector>.at(activation)(biased_product, activated);

}

void Layer::forward(const VectorBundle &prevVals) {

  VectorBundle output(weights.r, prevVals.c, 0);
  weights.mv2p(prevVals, output); // Forward the data

  output.addvh(biases); // Add the bias
	
  activatedm = output;
  apply_activation<VectorBundle>.at(activation)(output, activatedm);

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

  VectorBundle dl(W.c, delta.c, 0);
  W.mv2pt(delta, dl); // Derivative of the weights with respect to the loss in dl
	
  activate_gradient<VectorBundle>.at(activation)(activatedm, activatedm); 
  delta = activatedm * dl; // Derivative of the current layer

  dW.outer2(delta,prev);
  // Summing deltas over the batch to average later
  dw = dw + dW;
  Vector mdelta = delta.meanv();
  db = db + mdelta;

}

#if 0
// polymorphic: one version for the output alyer, and one for all others.
void Layer::backward_update( const VectorBundle &prev_wdelta, const VectorBundle& prev_activated ) {
  if (input_layer) {
    const VectorBundle& prev = input;
    dW.outer2(delta,prev);
  } else {
    const VectorBundle& prev = prev_activated;
    dW.outer2(delta,prev);
  }

  //  delta = a1a * wdelta;
  delta = activatedm * prev_wdelta; // Derivative of the current layer
  dw = dw + dW;
  Vector mdelta  = delta.meanv();
  db = db + mdelta;

  if (not input_layer) {
    wdelta = VectorBundle(weights.c, delta.c, 0);
    prev_weights.mv2pt(prev_delta, wdelta);
  }
};
#endif
