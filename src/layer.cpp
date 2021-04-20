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

#include "layer.h"
#include <iostream>

Layer::Layer() {};
Layer::Layer(int insize,int outsize)
  : weights( Matrix(outsize,insize,1) ),
    dw     ( Matrix(outsize,insize, 0) ),
	dW( Matrix(outsize,insize, 0) ),
    dw_velocity( Matrix(outsize,insize, 0) ),
    biases( Vector(insize, 1 ) ),
    biased_product( Vector(outsize, 0) ),
    activated( Vector(outsize, 0) ),
    d_activated( Vector(outsize, 0) ),
    biased_productm( VectorBatch(outsize,insize,0) ),
    activated_batch( VectorBatch(outsize,insize, 0) ),
    am1a( VectorBatch(outsize,insize, 0) ),
    d_activated_batch ( VectorBatch(outsize,insize, 0) ),
    db( Vector(insize, 0) ),
    delta_mean( Vector(insize, 0) ),
    dl( VectorBatch(insize, 0) ),
    db_velocity( Vector(insize, 0) ) {};


//codesnippet layerforward
void Layer::forward(const VectorBatch &prevVals) {

	VectorBatch output( prevVals.r, weights.c, 0 );
    prevVals.v2mp( weights, output );
	output.addh(biases); // Add the bias
    activated_batch = output;
    apply_activation<VectorBatch>.at(activation)(output, activated_batch);
}
//codesnippet end



void Layer::set_initial_deltas( const Matrix &dW, const Vector &delta ) {
  dw = dw + dW;
  db = db + delta;
};

/*
void Layer::set_recursive_deltas( Vector &delta, const Layer &next,const Layer &prev ) {
  backward(delta, next.weights, prev.activated);
};*/


void Layer::backward(VectorBatch &delta, const Matrix &W, const VectorBatch &prev) {

    //VectorBatch dl( delta.r, W.r ); 
    dl.resize( delta.batch_size(), W.rowsize() ); 
    delta.v2mtp( W, dl );

    activate_gradient<VectorBatch>.at(activation)(activated_batch, d_activated_batch); 
    delta = d_activated_batch * dl; // Derivative of the current layer

    update_dw(delta, prev);
}

void Layer::update_dw(VectorBatch &delta, VectorBatch prevValues) {
	prevValues.outer2(delta,dW);
	dw = dw + dW;
	delta_mean = delta.meanh();
	db = db + delta_mean;	
}

void Layer::backward_update( const VectorBatch &prev_wdelta, const VectorBatch& prev_output,bool input_layer ) {
  //  delta = a1a * wdelta;
  std::cout << activation << std::endl; 
  activate_gradient<VectorBatch>.at(activation)(activated_batch, am1a); 
  delta = am1a * prev_wdelta; // Derivative of the current layer

  dw = dw + dW;
  Vector mdelta  = delta.meanh();
  db = db + mdelta;

  //dW.outer2(delta,prev_output);
  delta.outer2(prev_output, dW);

  if (not input_layer) {
    //wdelta = VectorBatch(input_size(), delta.batch_size(), 0);
    //weights.mv2pt(delta, wdelta);
  	wdelta = VectorBatch( delta.c, weights.c, 0);
	delta.v2tmp( weights, wdelta );
  }
};
