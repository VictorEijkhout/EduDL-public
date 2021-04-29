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

#include <iostream>
using std::cout;
using std::endl;
#include <fstream>

#include <algorithm>
#include <string>
using std::string;

#include <cmath>

#include "vector.h"
#include "net.h"

Net::Net(int s) { // Input vector size
  this->inR = s;
  this->inC = 1;
  samples = 0;
}

Net::Net( const Dataset &data ) {
  this->inR = data.data_size(); //data.items.at(0).data.size();
  this->inC = 1;
  samples = 0;  
}

void Net::addLayer(int l, acFunc f) {
  addLayer( l,
	    apply_activation<VectorBatch>.at(f),
	    activate_gradient<VectorBatch>.at(f) );

  // int newR;
  // // For the first layer we need the input row size,
  // // for others we take the previous layer's row size
  // if (this->layers.empty()) {
  //     newR = this->inR; // Input's row size for the first layer
  // } else {
  //     newR = this->layers.back().output_size(); // Previous layer's row size
  // }

  // Layer layer(l, newR); // Initialize layer object and add the necessary parameters
    
  // layer.set_activation(f);                   // Activation function
  // this->layers.push_back(layer);          // New layer added

}
void Net::addLayer( int l,
		    std::function< void(const VectorBatch&,VectorBatch&) > apply_activation_batch,
		    std::function< void(const VectorBatch&,VectorBatch&) > activate_gradient_batch
		    ) {
  try {
    int newR;
    // For the first layer we need the input row size,
    // for others we take the previous layer's row size
    if (this->layers.empty()) {
      newR = this->inR; // Input's row size for the first layer
    } else {
      newR = this->layers.back().output_size(); // Previous layer's row size
    }
    Layer layer(newR, l); // Initialize layer object and add the necessary parameters
    layer.set_activation(apply_activation_batch,activate_gradient_batch);
    layer.layer_number = this->layers.size();
#ifdef DEBUG
    cout << "Creating layer " << layer.layer_number << ": "
	 << newR << "=>" << l << endl;
#endif
    this->layers.push_back(layer);
  } catch (std::string e ) {
    cout << "ERROR: <<" << e << ">> in adding layer " << l << endl;
  } catch (...) {
    throw( std::string("Error in addLayer") );
  }    
};


//codesnippet netforward
void Net::feedForward(const VectorBatch &input) {
  allocate_batch_specific_temporaries(input.batch_size());

  this->layers.front().forward(input); // Forwarding the input
  for (unsigned i = 1; i < layers.size(); i++) {
    this->layers.at(i).forward(this->layers.at(i - 1).activated_batch);
  }
}
//codesnippet end


void Net::show() {
  for (unsigned i = 0; i < layers.size(); i++) {
    cout << "Layer " << i << " weights" << endl;
    layers.at(i).weights.show();
  }
}

Categorization Net::output_vector() const {
  return Categorization( this->layers.back().activated ); // Return the final output
}

const VectorBatch &Net::outputs() const {
  return this->layers.back().activated_batch; // Return the final output
}


void Net::calculate_initial_delta(VectorBatch &input, VectorBatch &gTruth) {
  VectorBatch d_loss = d_lossFunction( gTruth, input);

  if (layers.back().activation == 2 ) { // Softmax derivative function
    Matrix jacobian( input.item_size(), input.item_size(), 0 );
    for(int i = 0; i < input.batch_size(); i++ ) {
      auto one_column = input.get_vector(i);
      jacobian = smaxGrad_vec( one_column );
      Vector one_vector( jacobian.rowsize(), 0 );
      Vector one_grad = d_loss.get_vectorObj(i);
      jacobian.mvp( one_grad, one_vector );
      layers.back().d_activated_batch.set_vector(one_vector,i);
    }
  }
  /* Will add the rest of the code here, not done yet
   */
}

void Net::backPropagate(const VectorBatch &input, const VectorBatch &gTruth) {
  VectorBatch delta = layers.back().activated_batch - gTruth;
  delta.scaleby( 1.f / gTruth.batch_size() );
  VectorBatch prev = layers.at(layers.size() - 2).activated_batch;
  //Matrix dW(delta.item_size(), prev.item_size(), 0);
	
  layers.back().update_dw(delta, prev);

  for (unsigned i = layers.size() - 2; i > 0; i--) {
    layers.at(i).backward
      ( layers.at(i+1).delta, layers.at(i+1).weights, layers.at(i-1).activated_batch);
  }
  layers.at(0).backward(layers.at(1).delta, layers.at(1).weights, input);
	
}

void Net::SGD(float lr, float momentum) {
  int samplesize = layers.at(0).activated_batch.batch_size();
  for (int i = 0; i < layers.size(); i++) {
    // Normalize gradients to avoid exploding gradients
    Matrix deltaW = layers.at(i).dw / samplesize;
    Vector deltaB = layers.at(i).db / samplesize;

    // Gradient descent
    if (momentum > 0.0) {
      layers.at(i).dw_velocity = momentum * layers.at(i).dw_velocity - lr * deltaW;
      //layers.at(i).weights = layers.at(i).weights + layers.at(i).dw_velocity;
      layers.at(i).weights.axpy( 1.f,layers.at(i).dw_velocity );
    } else {
      //layers.at(i).weights = layers.at(i).weights - lr * deltaW;
      layers.at(i).weights.axpy( -lr,deltaW );
    }

    layers.at(i).biases = layers.at(i).biases - lr * deltaB;

    // Reset the values of delta sums
    layers.at(i).dw.zeros();
    layers.at(i).db.zeros();
  }
}

void Net::RMSprop(float lr, float momentum) {
  for (int i = 0; i < layers.size(); i++) {
    // Get average over all the gradients
    Matrix deltaWsq = layers.at(i).dw;
    Vector deltaBsq = layers.at(i).db;
       	
    // Gradient step
    deltaWsq.square(); // dW^2
    deltaBsq.square(); // db^2
    // Sdw := m*Sdw + (1-m) * dW^2
		
    layers.at(i).dw_velocity = momentum * layers.at(i).dw_velocity + (1 - momentum) * deltaWsq;
    layers.at(i).db_velocity = momentum * layers.at(i).db_velocity + (1 - momentum) * deltaBsq;
		
    Matrix sqrtSdw = layers.at(i).dw_velocity;
    std::for_each(sqrtSdw.values().begin(), sqrtSdw.values().end(),
		  [](auto &n) { 
		    n = sqrt(n);
		    if(n==0) n= 1-1e-7;
		  });
    Vector sqrtSdb = layers.at(i).db_velocity;
    std::for_each(sqrtSdb.values().begin(), sqrtSdb.values().end(),
		  [](auto &n) { 
		    n = sqrt(n);
		    if(n==0) n= 1-1e-7;
		  });
		
    // W := W - lr * dW / sqrt(Sdw)
    layers.at(i).weights = layers.at(i).weights - lr * layers.at(i).dw / sqrtSdw;
    layers.at(i).biases = layers.at(i).biases - lr * layers.at(i).db / sqrtSdb;
			
    // Reset the values of delta sums
    layers.at(i).dw.zeros();
    layers.at(i).db.zeros();
  }
}

// this function no longer used
void Net::calcGrad(VectorBatch data, VectorBatch labels) {
  feedForward(data);
  backPropagate(data, labels);
}


void Net::train( const Dataset &data, int epochs, lossfn lossFuncName, int batchSize ) {

  cout << accuracy(data) << "\n";
  auto [trainSplit,testSplit] = data.split(0.95);
	
  const int Optimizer = optimizer();
  cout << "Optimizing with ";
  switch (Optimizer) {
  case sgd:  cout << "Stochastic Gradient Descent\n";  break;
  case rms:  cout << "RMSprop\n"; break;
  }

  lossFunction = lossFunctions.at(lossFuncName);
  d_lossFunction = d_lossFunctions.at(lossFuncName);
	
  std::vector<Dataset> batches = data.batch(batchSize);
  float lrInit = learning_rate();
  const float momentum_value = momentum();

  for (int i_epoch = 0; i_epoch < epochs; i_epoch++) {
    // Iterate through the entire dataset for each epoch
    cout << endl << "Epoch " << i_epoch+1 << "/" << epochs << endl;
    float current_learning_rate = lrInit; // Reset the learning rate to undo decay

    for (int j = 0; j < batches.size(); j++) {
      // Iterate through all batches within dataset
      auto& batch = batches.at(j);
#ifdef DEBUG
      cout << ".. batch " << j << "/" << batches.size() << " of size " << batch.size() << "\n";
#endif
      allocate_batch_specific_temporaries(batch.size());
      feedForward(batch.inputs());
      backPropagate(batch.inputs(),batch.labels());

      // User chosen optimizer
      current_learning_rate = current_learning_rate / (1 + decay() * j);
      optimize.at(Optimizer)(current_learning_rate, momentum_value); 
		
    }
    cout << " Loss: " << calculateLoss(data)
	 << " Accuracy: " << accuracy(data) << endl;
  }

}

/*
 * Resize temporaries to reflect current batch size
 */
void Net::allocate_batch_specific_temporaries(int batchsize) {
#ifdef DEBUG
  cout << "allocating temporaries for batch size " << batchsize << endl;
#endif
  for ( auto& layer : layers )
    layer.allocate_batch_specific_temporaries(batchsize);
}

/*!
 * Calculate the los function as sum of losses
 * of the individual data point.
 */
//codesnippet netloss
float Net::calculateLoss(const Dataset &testSplit) {

#ifdef DEBUG
  cout << "Loss calculation\n";
#endif
  allocate_batch_specific_temporaries(testSplit.inputs().batch_size());
  feedForward(testSplit.inputs());
  const VectorBatch &result = outputs();
  assert( result.notnan() );

  float loss = 0.0;
  for (int vec=0; vec<result.batch_size(); vec++) { // iterate over all items
    const auto& one_result = result.extract_vector(vec); // VLE figure out const span !!!
    auto tmp_labels =  testSplit.labels();
    auto one_label  = tmp_labels.get_vector(vec);
    assert( one_result.size()==one_label.size() );
    for (int i=0; i<one_result.size(); i++) { // Calculate loss of result
      auto this_label = one_label[i], this_result = one_result[i];
      assert( not std::isnan(this_label) );
      assert( not std::isnan(this_result) );
      auto oneloss = lossFunction( this_label, this_result );
      assert( not std::isnan(oneloss) );
      loss += oneloss;
    }
  }
  const int bs = result.batch_size();
  assert( bs>0 );
  auto scale = 1.f / static_cast<float>(bs);
  loss = -loss * scale;
    
  return loss;
}
//codesnippet end


#if 0
const auto& result_vals = result.vals_vector();
const auto& label_vals  = testSplit.labelBatch.vals_vector();
for (int j = 0; j < result.r * result.c; j++) {
  loss += lossFunction( label_vals[j], result_vals[j] );
 }
#else
#endif

float Net::accuracy( const Dataset &valSet ) {
#ifdef DEBUG
  cout << "Accuracy calculation\n";
#endif
  int correct = 0;
  int incorrect = 0;

  try {
    //      valSet.stack();
    auto [train,test_set] = valSet.split(0.95);
    assert( test_set.size()>0 );
    const auto& test_inputs = test_set.inputs();
    const auto& test_labels = test_set.labels();
    assert( test_inputs.batch_size()==test_labels.batch_size() );
    assert( test_inputs.batch_size()>0 );

    allocate_batch_specific_temporaries(test_inputs.batch_size());
    feedForward(test_inputs); // (valSet.dataBatch);
    const VectorBatch& output = outputs(); 
    assert( output.notnan() );

    for(int idx=0; idx < output.batch_size(); idx++ ) {
      Vector oneItem = output.get_vectorObj(idx);
      Categorization result( oneItem );
      result.normalize();
      if ( true ) { // valSet.items().at(idx).label.close_enough( result ) ) {
	correct++;
      } else {
	incorrect++;
      }
    }
    assert( correct+incorrect==test_set.size() );
  } catch (string e ) {
    cout << "ERROR: <<" << e << ">> in accuracy test" << endl;
    throw( string("Net::accuracy failed") );
  } catch (std::out_of_range) {
    cout << "Out of range error in accuracy test" << endl;
    throw( string("Net::accuracy failed") );
  } catch (...) {
    cout << "ERROR in accuracy test" << endl;
    throw( string("Net::accuracy failed") );
  }      
  float acc = static_cast<float>( correct ) / static_cast<float>( valSet.size() );
  return acc;
}



void Net::saveModel(std::string path){
  /*
    1. Size of matrix m x n, activation function
    2. Values of weight matrix
    3. Values of bias vector 
    4. Repeat for all layers
  */
  std::ofstream file;
  file.open( path, std::ios::binary );
	
  float temp;
  int no_layers = layers.size();
  file.write( reinterpret_cast<char *>(&no_layers), sizeof(no_layers) ); 
  for ( auto l : layers ) {
    int insize = l.input_size(), outsize = l.output_size();
    file.write(	reinterpret_cast<char *>(&outsize), sizeof(int) );
    file.write(	reinterpret_cast<char *>(&insize), sizeof(int) );
    file.write( reinterpret_cast<char *>(&l.activation), sizeof(int) );
		
    const auto& weights = l.weights;
    file.write(reinterpret_cast<const char *>(weights.data()), sizeof(float)*weights.nelements());
    // const float* weights_data = l.weights.data();
    // file.write(reinterpret_cast<char *>(&weights_data), sizeof(temp)*insize*outsize);

    const auto& biases = l.biases;
    file.write(reinterpret_cast<const char*>(biases.data()), sizeof(float) * biases.size());
    //file.write(reinterpret_cast<char *>(&l.biases.vals[0]), sizeof(temp) * l.biases.size());
  }
  cout << endl;

  file.close();
}

void Net::loadModel(std::string path){
  std::ifstream file(path);
  std::string buffer;
	
  int no_layers;
  file.read( reinterpret_cast<char *>(&no_layers), sizeof(no_layers) );
	
  float temp;
  layers.resize(no_layers);
  for ( int i=0; i < layers.size(); i++ ) {
    int insize,outsize;
    file.read( reinterpret_cast<char *>(&outsize), sizeof(int) );
    file.read( reinterpret_cast<char *>(&insize), sizeof(int) );
	
    file.read( reinterpret_cast<char *>(&layers[i].activation), sizeof(int) );

    layers[i].weights = Matrix( outsize, insize, 0 );
    float *w_data = layers[i].weights.data();
    file.read(reinterpret_cast<char *>( w_data ), //(&layers[i].weights.mat[0]), 
	      sizeof(temp) * insize*outsize);

    layers[i].biases = Vector( outsize, 0 );
    float *b_data = layers[i].biases.data();
    file.read(reinterpret_cast<char *>( b_data ), //(&layers[i].biases.vals[0]), 
	      sizeof(temp) * layers[i].biases.size());
  }
}


void Net::info() {
  cout << "Model info\n---------------\n";

  for ( auto l : layers ) {
    cout << "Weights: " << l.output_size() << " x " << l.input_size() << "\n";
    cout << "Biases: " << l.biases.size() << "\n";
		
    switch (l.activation) {
    case RELU: cout << "RELU\n"; break;
    case SIG: cout << "Sigmoid\n"; break;
    case SMAX: cout << "Softmax\n"; break;
    case NONE: break;
    }
    cout << "---------------\n";
  }

}

void loadingBar(int currBatch, int batchNo, float acc, float loss) {
  cout  << "[";
  int pos = 50 * currBatch/(batchNo-1);
  for (int k=0; k < 50; ++k) {
    if (k < pos) cout << "=";
    else if (k == pos) cout << ">";
    else cout << " ";
  }
  cout << "] " << int(float(currBatch)/float(batchNo-1)*100) << "% " << "loss: " << loss << " acc: " << acc << " \r";
  cout << std::flush;
}
