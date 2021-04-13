//
// Created by Ilknur on 17-Dec-20.
//

#ifndef SRC_LAYER_H
#define SRC_LAYER_H

#include <functional>

#include "vector.h"
#include "matrix.h"
#include "funcs.h"

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
    VectorBundle activated_batch,am1a,delta,wdelta;
    Vector d_activated; // for backpropagation
    VectorBundle biased_productm;
    VectorBundle d_activated_batch;
  
	Vector dl; // dloss
	Matrix dW; // dw calculated per layer
	Matrix dw;		// cumulative dw
    Matrix dw_velocity; // For SGD with Momentum, RMSprop
    Vector db_velocity;
    Vector db;		// cumulative deltas
	
	Vector delta_mean; // mean of the deltas used in batch training
public:

    int input_size() const { return weights.c; };
	int output_size() const { return weights.r; };
    void set_initial_deltas( const Matrix&, const Vector& );
    void set_recursive_deltas( Vector &, const Layer&,const Layer& );
    virtual void forward( const Vector &prevVals); // Virtual to support other types of layers later
    virtual void backward(Vector &delta, const Matrix &W, const Vector &prev);
    void backward_update( const VectorBundle&, const VectorBundle& ,bool=false );
   
   	void update_dw(VectorBundle &delta, VectorBundle prevValues);

	virtual void forward( const VectorBundle &prevVals);
    virtual void backward(VectorBundle &delta, const Matrix &W, const VectorBundle &prev);
		 
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

};


#endif //SRC_LAYER_H
