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
#include "trace.h"

#include <iostream>
using std::cout;
using std::endl;

Layer::Layer() {};
Layer::Layer(int insize,int outsize)
  : weights( Matrix(outsize,insize,1) ),
    dw     ( Matrix(outsize,insize, 0) ),
	//dW( Matrix(outsize,insize, 0) ),
    dw_velocity( Matrix(outsize,insize, 0) ),
    biases( Vector(outsize, 1 ) ),
    // biased_product( Vector(outsize, 0) ),
    activated( Vector(outsize, 0) ),
    d_activated( Vector(outsize, 0) ),
    delta( VectorBatch(outsize,1) ),
    // biased_productm( VectorBatch(outsize,insize,0) ),
    //    activated_batch( VectorBatch(outsize,1, 0) ),
    //    d_activated_batch ( VectorBatch(outsize,insize, 0) ),
    db( Vector(insize, 0) ),
    // delta_mean( Vector(insize, 0) ),
    dl( VectorBatch(insize, 1) ),
    db_velocity( Vector(insize, 0) ) {};

/*
 * Resize temporaries to reflect current batch size
 */
void Layer::allocate_batch_specific_temporaries(int batchsize) {
  const int insize = weights.colsize(), outsize = weights.rowsize();

  activated_batch.allocate( batchsize,outsize );
  d_activated_batch.allocate( batchsize,outsize );
  dl.allocate( batchsize, outsize );
  delta.allocate( batchsize,outsize );
};

void Layer::set_activation(acFunc f) {
  activation = f;
  apply_activation_batch  = apply_activation<VectorBatch>.at(f);
  activate_gradient_batch = activate_gradient<VectorBatch>.at(f);
};

void Layer::set_activation
( std::function< void(const VectorBatch&,VectorBatch&) > apply,
  std::function< void(const VectorBatch&,VectorBatch&) > activate ) {
  activation = acFunc::RELU;
  apply_activation_batch  = apply;
  activate_gradient_batch = activate;
};

void Layer::set_uniform_weights(float v) {
  for ( auto& e : weights.values() )
    e = v;
};

void Layer::set_uniform_biases(float v) {
  for ( auto& e : biases.values() )
    e = v;
};

//codesnippet layerforward
void Layer::forward(const VectorBatch &prevVals) {
#ifdef DEBUG
  cout << "Forward layer " << layer_number
       << ": " << input_size() << "->" << output_size() << endl;
#endif

    assert( prevVals.notnan() ); assert( prevVals.notinf() );
    prevVals.v2mp( weights, activated_batch );
    assert( activated_batch.notnan() ); assert( activated_batch.notinf() );

    activated_batch.addh(biases); // Add the bias
    assert( activated_batch.notnan() ); assert( activated_batch.notinf() );

    apply_activation_batch(activated_batch, activated_batch);
    assert( activated_batch.notnan() ); assert( activated_batch.notinf() );
}
//codesnippet end

void Layer::backward
    (const VectorBatch &prev_delta, const Matrix &W, const VectorBatch &prev_output) {

  prev_delta.v2mtp( W, dl );
  activate_gradient_batch(activated_batch, d_activated_batch); 
  delta.hadamard( d_activated_batch,dl ); // Derivative of the current layer
  //  update_dw(delta, prev_output);
  prev_output.outer2( delta, dw );
  weights.axpy( 1.,dw );
  db = delta.meanh();
  biases.add( db );
}

void Layer::update_dw( const VectorBatch &delta, const VectorBatch& prevValues) {
  prevValues.outer2( delta, dw );
  if (trace_scalars())
    cout << "dw: " << dw.normf() << "\n";
  weights.axpy( 1.,dw );
  db = delta.meanh();
  biases.add( db );
}

