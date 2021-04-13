//
// Created by Ilknur on 29-Nov-20.
//

#ifndef CODE_NET_H
#define CODE_NET_H

#include <vector>
#include "vector.h"
#include "matrix.h"
#include "dataset.h"
#include "layer.h"
#include <cmath>

enum opt{sgd, rms}; // Gradient descent, RMSprop
enum lossfn{cce, mse}; // categorical cross entropy, mean squared error

class Net {
private:
    int inR; // input dimensions
    int inC;
    int samples;
    std::vector<Layer> layers;
    std::function<float( const float& groundTruth, const float& result)> lossFunction;
	std::function<VectorBundle( VectorBundle& groundTruth, VectorBundle& result )> d_lossFunction;
public:
    Net(int s); // input shape
    Net( const Dataset &d );
    void addLayer(int l, acFunc activation); // length of the dense layer
    void show(); // Show all weights
    Categorization output_vector() const;
    VectorBundle &output_mat();

    void feedForward( const Vector& );
    void feedForward( const VectorBundle& );

    void calcGrad(Dataset data);
    void calcGrad(VectorBundle data, VectorBundle labels);

    void backPropagate(const Vector &input, const Vector &gTruth);
    void backPropagate(const VectorBundle &input, const VectorBundle &gTruth);
	
	void calculate_initial_delta( VectorBundle& result, VectorBundle& gTruth);

    void SGD(float lr, float momentum);
    void RMSprop(float lr, float momentum);

  /*
   * Various settings
   */
private:
  float _lr{0.05};
public:
  void set_learning_rate(float lr) { _lr=lr; };
  float learning_rate() const { return _lr; };
private:
  float _decay{0.05}; // high decay causes very small training
public:
  void set_decay(float d) { _decay = d; };
  float decay() const { return _decay; };
private:
  float _momentum{0.0}; // 0.9 works well
public:
  void set_momentum(float m) { _momentum = m; };
  float momentum() const { return _momentum; };
private:
  int _optimizer;
public:
  void set_optimizer(int m) { _optimizer = m; };
  int optimizer() const { return _optimizer; };
  std::vector< std::function< void(float lr, float momentum) > > optimize{
    [this] ( float lr, float momentum ) { SGD(lr, momentum); },
    [this] ( float lr, float momentum ) { RMSprop(lr, momentum); }
  };
	
    void train(Dataset &trainData, Dataset &testData, int epochs, lossfn lossFunc, int batchSize);
#if MPINN
	void trainmpi(Dataset &trainData, Dataset &testData, float lr, int epochs, opt Optimizer, lossfn lossFunc, int batchSize, float momentum = 0.0, float decay = 0.0);
#endif
	float calculateLoss(Dataset &testSplit);
    float accuracy(Dataset& valSet);

    inline static std::vector< std::function< float( const float& groundTruth, const float& result) > > lossFunctions{
	[] ( const float &gT, const float &result ) { return gT * log(result); }, // Categorical Cross Entropy
	[] ( const float &gT, const float &result ) { return pow(gT - result, 2); }, // Mean Squared Error
	};

	inline static std::vector< std::function< VectorBundle( VectorBundle& groundTruth, VectorBundle& result) > > d_lossFunctions{
	[] ( VectorBundle &gT, VectorBundle &result ) { return -gT/result/result.bundle_size(); },
	[] ( VectorBundle &gT, VectorBundle &result ) { return -2 * ( gT-result)/result.bundle_size(); },
	};

	void saveModel(std::string path);
	void loadModel(std::string path);
	
	void info();
};

void loadingBar(int currBatch, int batchNo, float acc, float loss);

#endif //CODE_NET_H
