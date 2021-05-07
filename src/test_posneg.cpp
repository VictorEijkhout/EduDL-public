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
#include <chrono>
#include <vector>
#include <time.h>

#include "cxxopts.hpp"

#include "net.h"
#include "dataset.h"
#include "vector.h"
#include "funcs.h"
#include "trace.h"

using namespace std;

int main(int argc,char **argv){

  try {
    using myclock = std::chrono::high_resolution_clock;

    // IM attempt to use cxxopts to specify location
    cxxopts::Options options("EduDL", "FFNNs w BLIS");

    options.add_options()
      ("h,help","usage information")
      ("l,levels", "Number of levels in the network",cxxopts::value<int>()->default_value("2"))
      ("s,sizes","Sizes of the levels",cxxopts::value<std::vector<int>>())
      ("o,optimizer", "Optimizer to be used, 0: SGD, 1: RMSprop",cxxopts::value<int>()->default_value("0"))
      ("e,epochs", "Number of epochs to train the network", cxxopts::value<int>()->default_value("1"))
      ("r,learningrate", "Learning rate for the optimizer", cxxopts::value<float>()->default_value("0.001"))
      ("b,batchsize", "Batch size for the training data", cxxopts::value<int>()->default_value("5"))
      ("t,tracing","Level of tracing: 0=default 1=scalars 2=arrays",cxxopts::value<int>()->default_value("0"))
      ;
		
    auto result = options.parse(argc,argv);
    if (result.count("help")) {
      std::cout << options.help() << std::endl;
      return 1;
    }

    std::vector<int> level_sizes{12,12};

    if (result.count("sizes")) {
      level_sizes = result["s"].as< std::vector<int> >();
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
    Dataset data(2);
    for (float item=-99.5; item<100; item+=1) {
      if (item<0) {
	// negative numbers are yes/no
	dataItem thisitem{std::vector<float>{item},std::vector<float>{1,0}};
	data.push_back(thisitem);
      } else {
	// positive numbers are no/yes
	dataItem thisitem{std::vector<float>{item},std::vector<float>{0,1}};
	data.push_back(thisitem);
      }
    }
    cout << "Dataset size: " << data.size() << endl; // Show size

    cout << "reinstate shuffling\n"; //    data.shuffle();

    Net test_net( data );
    if (level_sizes.size()==2) {
      test_net.addLayer(16, SIG); // RELU);
      test_net.addLayer(2, SIG); // SMAX );
    } else {
      for ( auto level_size : level_sizes ) {
	std::cout << "Adding level of size " << level_size << std::endl;
	test_net.addLayer(level_size,SIG); // RELU);
      }
      test_net.addLayer(2, SIG); // SMAX );
    }

    test_net.set_learning_rate(lr);
    test_net.set_decay(0.9);
    test_net.set_momentum(0.9);
    test_net.set_optimizer(network_optimizer);
    test_net.set_lossfunction(mse);

    auto start_time = myclock::now();
    auto [train_data,test_data] = data.split(0.9);
    cout << "Initial accuracy: " << test_net.accuracy(test_data) << "\n";

    test_net.train(train_data,test_data, epochs, batchSize);
    auto duration = myclock::now()-start_time;
    auto microsec_duration = std::chrono::duration_cast<std::chrono::microseconds>(duration);
    std::cout << "Final Accuracy over test data: " << test_net.accuracy(data) << "\n"
	      << "    attained in " << microsec_duration.count() << "usec" << "\n";
	
    //test_net.saveModel("weights.bin");
    test_net.info();

  } catch ( string e ) {
    cout << "Error <<" << e << ">> \n";
  } catch ( std::out_of_range ) {
    cout << "Out of range error\n";
  } catch ( ... ) {
    cout << "Uncaught exception\n";
  }

    return 0;
}
