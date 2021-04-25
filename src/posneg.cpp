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

using namespace std;

int main(int argc,char **argv){
  //    srand(time(NULL));

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
  cout << data.size() << endl; // Show size

  data.shuffle();

  data.stack();

  // std::cout << "Data: "  << data.dataBatch.r << " " << data.dataBatch.c << std::endl;
  // std::cout << "Label: " << data.labelBatch.r << " " << data.labelBatch.c << std::endl;
	
  Net test_net( data );
    
  if (level_sizes.size()==2) {
    test_net.addLayer(16, RELU);
    test_net.addLayer(2, SMAX);
  } else {
    for ( auto level_size : level_sizes ) {
      std::cout << "Adding level of size " << level_size << std::endl;
      test_net.addLayer(level_size,RELU);
      //test_net.addLayer(level_size,relu_io,reluGrad_io );
    }
    test_net.addLayer(2, SMAX );
    //test_net.addLayer(2, softmax_io,softmaxGrad_io );
  }
  test_net.set_learning_rate(lr);
  test_net.set_decay(0.9);
  test_net.set_momentum(0.9);
  test_net.set_optimizer(network_optimizer);

  auto [trainSplit,testSplit] = data.split(0.95);
	
  std::cout << test_net.accuracy(data) << "\n";
  auto start_time = myclock::now();
  test_net.train(trainSplit,testSplit,
		 epochs, cce, batchSize);
  auto duration = myclock::now()-start_time;
  auto microsec_duration = std::chrono::duration_cast<std::chrono::microseconds>(duration);
  std::cout << "Final Accuracy over all data: " << test_net.accuracy(data) << "\n"
	    << "    attained in " << microsec_duration.count() << "usec" << "\n";
	
  //test_net.saveModel("weights.bin");
  test_net.info();

  return 0;
}
