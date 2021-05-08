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
  //    srand(time(NULL));

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
     * Input data set: 
     * y = 2 x + 1
     */
    cout << "Creating data set\n";
    Dataset data(1);
    for (int input=0; input<5; input++) {
      std::vector<float> i(1,1.f*input), o(1,2*input+1.);
      //inputs.add_vector(i); outputs.add_vector(o);
      dataItem thisitem{i,o};
      data.push_back(thisitem);
    }
    cout  << "Number of data points: " << data.size() << endl; // Show size
    //    data.stack();

    {
      /* 
       * First test a perfect net
       */
      Net test_net( data );
      test_net.addLayer(1,NONE);
      test_net.set_uniform_weights(1.f);
      test_net.set_uniform_biases(0.f);
      test_net.set_lossfunction(mse);
	
      auto loss = test_net.calculateLoss(data);
      cout << " Loss: " << loss << endl;
      auto acc = test_net.accuracy(data);
      std::cout << "Accuracy: " << acc << "\n";
      test_net.info();

      auto [train_data,test_data] = data.split(0.9);
      cout << "Initial accuracy: " << test_net.accuracy(test_data) << "\n";
      test_net.train(train_data,test_data, epochs, batchSize);
      cout << "Final Accuracy over test data: " << acc << "\n";

    }

  } catch ( string e ) {
    cout << "Error <<" << e << ">> \n";
  } catch ( std::out_of_range ) {
    cout << "Out of range error\n";
  } catch ( ... ) {
    cout << "Uncaught exception\n";
  }

    return 0;
}
