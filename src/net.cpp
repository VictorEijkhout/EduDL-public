//
// Created by Ilknur on 29-Nov-20.
//

#include "vector.h"
#include "net.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>

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
  int newR;
  // For the first layer we need the input row size,
  // for others we take the previous layer's row size
  if (this->layers.empty()) {
    newR = this->inR; // Input's row size for the first layer
  } else {
    newR = this->layers.back().weights.r; // Previous layer's row size
  }

  Layer layer(newR, l); // Initialize layer object and add the necessary parameters
    
  layer.activation = f;                   // Activation function
  this->layers.push_back(layer);          // New layer added

}


//Vector Net::feedForward(Vector input) {
void Net::feedForward( const Vector &input ) {
  this->layers.front().forward(input); // Forwarding the input
  for (unsigned i = 1; i < layers.size(); i++) {
    this->layers.at(i).forward(this->layers.at(i - 1).activated);
  }

}

void Net::feedForward(const VectorBundle &input) {
  this->layers.front().forward(input); // Forwarding the input

  for (unsigned i = 1; i < layers.size(); i++) {
    this->layers.at(i).forward(this->layers.at(i - 1).activatedm);
  }
}


void Net::show() {
  for (unsigned i = 0; i < layers.size(); i++) {
    std::cout << "Layer " << i << " weights" << std::endl;
    layers.at(i).weights.show();
  }
}

Categorization Net::output_vector() const {
  return Categorization( this->layers.back().activated ); // Return the final output
}

VectorBundle &Net::output_mat() {
  return this->layers.back().activatedm; // Return the final output
}

void Net::backPropagate(const Vector &input, const Vector &gTruth) {
  Vector delta = layers.back().activated - gTruth;
  // This works for logistic regression and softmax classification but will be abstracted
  Matrix dW(delta.size(), layers.at(layers.size()-2).activated.size(), 0);
  dW.outerProduct(delta, layers.at(layers.size() - 2).activated);

  layers.back().set_initial_deltas( dW,delta );

  for (unsigned i = layers.size() - 2; i > 0; i--) {
    layers.at(i).set_recursive_deltas(delta, layers.at(i + 1), layers.at(i - 1) );
  }
  layers.at(0).backward(delta, layers.at(1).weights, input);

}

#if 1
void Net::backPropagate(const VectorBundle &input, const VectorBundle &gTruth) {
  VectorBundle delta = layers.back().activatedm - gTruth;

  VectorBundle prev = layers.at(layers.size() - 2).activatedm;
  Matrix dW(delta.r, prev.r, 0);
  dW.outer2( delta, prev );

  layers.back().dw = layers.back().dw + dW;
  Vector mdelta = delta.meanv();
  layers.back().db = layers.back().db + mdelta;

  for (unsigned i = layers.size() - 2; i > 0; i--) {
    layers.at(i).backward(delta, layers.at(i + 1).weights, layers.at(i - 1).activatedm);
  }
  layers.at(0).backward(delta, layers.at(1).weights, input);

}
#else
void Net::backPropagate(const VectorBundle &input, const VectorBundle &gTruth) {

  for (int i = layers.size()-1; i >=0; i--) {

    auto& this_layer = layers.at(i);
    if (i==layers.size()-1) {
      this_layer.backward_update( this_layer.activatedm - gTruth )
	} else {
      auto& prev_layer = layers.at(i+1);
      this_layer.backward_update( prev_layer.wdelta,prev_layer.activated );
    }

    // // apply sigmoid
    // this_layer.activate_gradient<VectorBundle>
    //   .at(this_layer.activation)(this_layer.activatedm, this_layer.activatedm); 
  }
#endif

  void Net::SGD(float lr, float momentum) {
    int samplesize = layers.at(0).activatedm.c;
    for (int i = 0; i < layers.size(); i++) {
      // Normalize gradients to avoid exploding gradients
      Matrix deltaW = layers.at(i).dw / samplesize;
      Vector deltaB = layers.at(i).db / samplesize;

      // Gradient descent
      if (momentum > 0.0) {
	layers.at(i).dw_velocity = momentum * layers.at(i).dw_velocity - lr * deltaW;
	layers.at(i).weights = layers.at(i).weights + layers.at(i).dw_velocity;
      } else {
	layers.at(i).weights = layers.at(i).weights - lr * deltaW;
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
      std::for_each(sqrtSdw.mat.begin(), sqrtSdw.mat.end(),
		    [](auto &n) { 
		      n = sqrt(n);
		      if(n==0) n= 1-1e-7;
		    });
      Vector sqrtSdb = layers.at(i).db_velocity;
      std::for_each(sqrtSdb.vals.begin(), sqrtSdb.vals.end(),
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

  void Net::calcGrad(VectorBundle data, VectorBundle labels) {
    feedForward(data);
    backPropagate(data, labels);
  }


  void Net::train(Dataset &data, Dataset &testData, int epochs, lossfn lossFuncName, int batchSize ) {

    const int Optimizer = optimizer();
    std::cout << "Optimizing with ";
    switch (Optimizer) {
    case sgd:  std::cout << "Stochastic Gradient Descent\n";  break;
    case rms:  std::cout << "RMSprop\n"; break;
    }


    lossFunction = lossFunctions.at(lossFuncName);
    int ssize = batchSize;//data.items.size();

    std::vector<Dataset> batches = data.batch(ssize);
    for (int i = 0; i < batches.size(); i++) {
      batches.at(i).stack(); // Put batch items into one matrix
    }
    float lrInit = learning_rate();
    const float momentum_value = momentum();
    for (int i_epoch = 0; i_epoch < epochs; i_epoch++) {
      // Iterate through the entire dataset for each epoch
      std::cout << std::endl << "Epoch " << i_epoch+1 << "/" << epochs;
      float current_learning_rate = lrInit; // Reset the learning rate to undo decay

      for (int j = 0; j < batches.size(); j++) {
	// Iterate through all batches within dataset
		
        // Calculate gradient at current batch
        calcGrad(batches.at(j).dataBatch, batches.at(j).labelBatch);
        
        // User chosen optimizer
        current_learning_rate = current_learning_rate / (1 + decay() * j);
        optimize.at(Optimizer)(current_learning_rate, momentum_value); 
		
      }
      std::cout << " Loss: " << calculateLoss(testData)
		<< " Accuracy: " << accuracy(testData) << std::endl;
    }
    std::cout << std::endl;
  }

  float Net::calculateLoss(Dataset &testSplit) {
    testSplit.stack();
    feedForward(testSplit.dataBatch);
    const VectorBundle &result = output_mat();

    float loss = 0.0;
#if 0
    const auto& result_vals = result.vals_vector();
    const auto& label_vals  = testSplit.labelBatch.vals_vector();
    for (int j = 0; j < result.r * result.c; j++) {
      loss += lossFunction( label_vals[j], result_vals[j] );
    }
#else
    for (int vec=0; vec<result.bundle_size(); vec++) {
      const auto& one_result = result.get_vector(vec);
      const auto& one_label  = testSplit.labelBatch.get_vector(vec);
      assert( one_result.size()==one_label.size() );
      for (int i=0; i<one_result.size(); i++)
	loss += lossFunction( one_label[i], one_result[i] );
    }
#endif

    loss = -loss / (float) testSplit.labelBatch.c;
    
    return loss;
  }

  float Net::accuracy(Dataset &valSet) {
    int correct = 0;
    int incorrect = 0;

    for ( auto &valSetItem : valSet.items() ) { 
      feedForward(valSetItem.data);
      Categorization result( output_vector() );
      result.normalize();
      if ( valSetItem.label.close_enough( result ) ) {
	correct++;
      } else {
	incorrect++;
      }
    }
    assert( correct+incorrect==valSet.size() );
    float acc = (float) correct / (float) valSet.size();
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
      file.write(	reinterpret_cast<char *>(&l.weights.r), sizeof(int) );
      file.write(	reinterpret_cast<char *>(&l.weights.c), sizeof(int) );
      file.write( reinterpret_cast<char *>(&l.activation), sizeof(int) );

      file.write(reinterpret_cast<char *>(&l.weights.mat[0]), sizeof(temp)*l.weights.r*l.weights.c);
      file.write(reinterpret_cast<char *>(&l.biases.vals[0]), sizeof(temp) * l.biases.size());
    }
    std::cout << std::endl;

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
      file.read( reinterpret_cast<char *>(&layers[i].weights.r), sizeof(int) );
      file.read( reinterpret_cast<char *>(&layers[i].weights.c), sizeof(int) );
	
      file.read( reinterpret_cast<char *>(&layers[i].activation), sizeof(int) );

      layers[i].weights = Matrix( layers[i].weights.r, layers[i].weights.c, 0 );
      file.read(reinterpret_cast<char *>(&layers[i].weights.mat[0]), 
		sizeof(temp) * layers[i].weights.r * layers[i].weights.c);

      layers[i].biases = Vector( layers[i].weights.r, 0 );
      file.read(reinterpret_cast<char *>(&layers[i].biases.vals[0]), 
		sizeof(temp) * layers[i].biases.size());
    }
  }


  void Net::info() {
    std::cout << "Model info\n---------------\n";

    for ( auto l : layers ) {
      std::cout << "Weights: " << l.weights.r << " x " << l.weights.c << "\n";
      std::cout << "Biases: " << l.biases.size() << "\n";
		
      switch (l.activation) {
      case RELU: std::cout << "RELU\n"; break;
      case SIG: std::cout << "Sigmoid\n"; break;
      case SMAX: std::cout << "Softmax\n"; break;
      case NONE: break;
      }
      std::cout << "---------------\n";
    }

  }

  void loadingBar(int currBatch, int batchNo, float acc, float loss) {
    std::cout  << "[";
    int pos = 50 * currBatch/(batchNo-1);
    for (int k=0; k < 50; ++k) {
      if (k < pos) std::cout << "=";
      else if (k == pos) std::cout << ">";
      else std::cout << " ";
    }
    std::cout << "] " << int(float(currBatch)/float(batchNo-1)*100) << "% " << "loss: " << loss << " acc: " << acc << " \r";
    std::cout << std::flush;
  }
