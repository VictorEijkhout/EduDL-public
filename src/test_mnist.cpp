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

/*
 * A simple test neural network
 * 
 * Test data set: http://cis.jhu.edu/~sachin/digit/digit.html
 */

#include <iostream>
using std::cout;
using std::endl;
#include <chrono>
#include <string>
using std::string;
#include <vector>
using std::vector;
#include <time.h>

#include "cxxopts.hpp"

#include "net.h"
#include "dataset.h"
#include "vector.h"
#include "trace.h"

using namespace std;
static int trace_level;

#if 1
int main(int argc,char **argv){
  try {
    using myclock = std::chrono::high_resolution_clock;

    // IM attempt to use cxxopts to specify location
    cxxopts::Options options("EduDL", "FFNNs w BLIS");

    options.add_options()
      ("h,help","usage information")
      ("d,dir", "Dataset directory",cxxopts::value<std::string>())
      ("l,levels", "Number of levels in the network",cxxopts::value<int>()->default_value("2"))
      ("s,sizes","Sizes of the levels",cxxopts::value<std::vector<int>>())
      ("o,optimizer", "Optimizer to be used, 0: SGD, 1: RMSprop",cxxopts::value<int>()->default_value("0"))
      ("e,epochs", "Number of epochs to train the network", cxxopts::value<int>()->default_value("1"))
      ("r,learningrate", "Learning rate for the optimizer", cxxopts::value<float>()->default_value("0.001"))
      ("b,batchsize", "Batch size for the training data", cxxopts::value<int>()->default_value("256"))
      ("t,tracing","Level of tracing: 0=default 1=scalars 2=arrays",cxxopts::value<int>()->default_value("0"))
	  ;
		
    auto result = options.parse(argc,argv);
    if (result.count("help")) {
      cout << options.help() << endl;
      return 1;
    }

    std::vector<int> level_sizes{12,12};

    if (result.count("sizes")) {
      level_sizes = result["s"].as< std::vector<int> >();
      if (result.count("levels")) {
	cout << "Option for number of levels ignored when level sizes are given" << endl;
      }
    } else if (result.count("levels")) {
      int number_of_levels = result["l"].as<int>();
      level_sizes = std::vector<int>(number_of_levels,12);
    }
	
    set_trace_level( result["t"].as<int>() );
    int network_optimizer = result["o"].as<int>();
    int epochs = epochs = result["e"].as<int>();
    float lr = result["r"].as<float>();
    int batchSize = result["b"].as<int>();

    /*
     * Input data set handling
     */
    if (!result.count("dir")) {
      cout << "Must specify directory with -d/--dir option" << endl;
      return 1;
    }
    string mnist_loc = result["dir"].as<string>();
    Dataset data;
    data.readTest(mnist_loc.data()); // Placed MNIST in a neighbor directory

    // Parent
    //   |__mnist
    //   |      |_data0
    //   |      |_...
    //   |      |_data9
    //   |__src
    //          |_test.cpp  [You are here]


    cout << "Dataset size: " << data.size() << endl; // Show size

    //data.shuffle();

    Net test_net(data); 
    if (level_sizes.size()==2) {
      test_net.addLayer(16, SIG ); //RELU);
      test_net.addLayer(10, SIG ); //SMAX);
    } else {
      for ( auto level_size : level_sizes ) {
	cout << "Adding level of size " << level_size << endl;
	test_net.addLayer(level_size,SIG ); //RELU);
      }
      test_net.addLayer(10, SIG ); //SMAX);
    }

    test_net.set_learning_rate(lr);
    test_net.set_decay(0.0);
    test_net.set_momentum(0.9);
    test_net.set_optimizer(network_optimizer);
      test_net.set_uniform_weights(.5f);
      test_net.set_uniform_biases(.1f);
      test_net.set_lossfunction(mse);

    /*
     * Train / Test 
     */
    auto start_time = myclock::now();
    auto [train_data,test_data] = data.split(0.9);
    cout << "Initial accuracy: " << test_net.accuracy(test_data) << "\n";

    test_net.train(train_data,test_data, epochs, batchSize);
    auto duration = myclock::now()-start_time;

    int
      microsec_duration = std::chrono::duration_cast<std::chrono::microseconds>(duration).count(),
      seconds = microsec_duration/1000000,
      micros  = microsec_duration - 1000000*seconds;
    while (micros>=1000) micros /= 10;
    auto acc = test_net.accuracy(test_data);
    cout << "Final Accuracy over test data: " << acc << "\n"
	 << "    attained in " << seconds << "." << micros << " sec" << "\n";
	
    test_net.saveModel("weights.bin");
    test_net.info();

  } catch ( string e ) {
    cout << "ERROR <<" << e << ">>\n";
  } catch ( std::out_of_range ) {
    cout << "Uncaught out of range error\n";
  } catch ( ... ) {
    cout << "Uncaught exception\n";
  }

    return 0;
}
#else
int main(){
	
	VectorBatch a(3,4,1); // Two feature vectors, each vector sized 4
	VectorBatch b(3,5,1);
	a.show();

	Net model(4);
	model.addLayer(2,RELU);
	model.addLayer(5,SMAX);
	
	model.show();
	model.feedForward(a);
	model.backPropagate(a, b);	

    return 0;
}
#endif
