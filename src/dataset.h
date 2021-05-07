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

#ifndef CODE_DATASET_H
#define CODE_DATASET_H
//#include "matrix.h"
#include "vector2.h"
#include "vector.h"
#include <vector>
#include <iostream>

class dataItem{
public: // should really be done through friends private:
    Vector data; // Data matrix
  //Vector label; // Label
    Categorization label; // Label
public:
  dataItem( Vector data,/* Vector */ Categorization label)
    : data(data),label(label) {};
  dataItem( float data,Categorization label)
    : data( std::vector<float>(data) ),label(label) {};
  dataItem( std::vector<float> indata,std::vector<float> outdata )
    : data( Vector(indata) ),label( Categorization(outdata) ) {};
  int data_size() const { return data.size(); };
  const std::vector<float>& data_values() const { return data.values(); };
  int label_size() const { return label.size(); };
  const std::vector<float>& label_values() const { return label.probabilities(); };
};

class Dataset {
private:
  int nclasses{0};
  //  std::vector<dataItem> _items; // All the data: {data, label}
private:
    VectorBatch dataBatch;
    VectorBatch labelBatch;
  int lowerbound{0},number{0};
public:
  Dataset() {};
  Dataset( int n );
  Dataset( std::vector<dataItem> );

  void set_lowerbound( int b );
  void set_number( int b );
  void push_back(dataItem it);
  int size() const;
  int data_size() const;
  int label_size() const;

  dataItem item(int i) const;
  //  const dataItem& at(int i) const { return _items.at(i); };
  //  dataItem& at(int i) { return _items.at(i); };
  const Vector& data(int i) const;
  const std::vector<float> data_vals(int i) const;
  const std::vector<float> label_vals(int i) const;
  std::vector<float> stacked_data_vals(int i) const;
  std::vector<float> stacked_label_vals(int i) const;
  //  const Categorization& label(int i) const { return _items.at(i).label; };

  //  const std::vector<dataItem>& items() const { return _items; };
  //const Vector& label(int i) const { return items.at(i).label; };

    std::string path; // Path of the dataset
public:
  const auto& inputs() const { return dataBatch; };
  const auto& labels() const { return labelBatch; };

    int readTest(std::string dataPath); // Read modified MNIST Dataset
    void shuffle(); // Mix the dataset
    std::vector<Dataset> batch(int n) const; // Divides the dataset into n batches
    void stack();
    std::pair<Dataset,Dataset> split(float trainFraction) const; // Train-test split
};


#endif //CODE_DATASET_H
