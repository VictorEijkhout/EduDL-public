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

#include "dataset.h"
#include <iostream>
#include <string>
#include <vector>
using std::vector;

#include <cstdio>
#include <algorithm>
#include <random>

#define IMSIZE 28

Dataset::Dataset( std::vector<dataItem> v )
  : _items(v) {
  if (v.size()>0)
    nclasses = v.at(0).label_size();
};

int Dataset::size() const {
  return _items.size();
}

/*!
 * What is the size of the feature vector in this dataset?
 */
int Dataset::data_size() const { return _items.at(0).data_size(); };
/*!
 * What is the number of categories in the labels of this dataset?
 */
int Dataset::label_size() const { return _items.at(0).label_size(); };

/*!
 * Get the i-th data object 
 */
const Vector& Dataset::data(int i) const {
  return _items.at(i).data;
};
/*!
 * Get the features of i-th data object 
 */
const vector<float>& Dataset::data_vals(int i) const {
  return _items.at(i).data_values();
};
//! Same, of the stacked object
vector<float> Dataset::stacked_data_vals(int i) const {
  //return dataBatch.get_col(i);
  return dataBatch.get_row(i);
};

/*!
 * Get the categorization of i-th data object 
 */
const vector<float>& Dataset::label_vals(int i) const {
  return _items.at(i).label_values();
};
//! Same, of the stacked object
vector<float> Dataset::stacked_label_vals(int i) const {
  //return labelBatch.get_col(i);
  return labelBatch.get_row(i);
};

/*!
 * Add a new data item, and check its consistency with previous items
 */
void Dataset::push_back(dataItem it) {
  if (nclasses>0 and it.label_size()!=nclasses)
    throw("number of classes mismatch");
  if (nclasses==0)
    nclasses = it.label_size();
  _items.push_back(it);
};

int Dataset::readTest(std::string dataPath) {
  /*
   * This reader is specifically for a modified MNIST dataset which
   * does not include the file header, metadata, etc.
   * Link to the dataset: http://cis.jhu.edu/~sachin/digit/digit.html
   * I chose this dataset for now to make it easy to read the data;
   * in later iterations I will generalize the read function, maybe OpenCV support
   */

  FILE *file;
  std::string fileName;
  uint8_t temp[IMSIZE * IMSIZE]; // Image buffer to read data into
  for (int dataid = 0; dataid < 10; dataid++) {
    fileName = dataPath + "/data" + std::to_string(dataid); // Put together the path
    file = fopen(fileName.c_str(), "r");


    if (!file) { // File checking
      std::cout << "Error opening file" << std::endl;
      return -2; // Arbitrary error code
    } 
    for (int k = 0; k < 1000; k++) {
      Vector imageVec(IMSIZE * IMSIZE, 0); // initialize matrix to be read into
      fread(temp, 1, IMSIZE * IMSIZE, file); // Read 28*28 into buffer

      for (int i = 0; i < IMSIZE; i++) {
	for (int j = 0; j < IMSIZE; j++) {
	  // Transfer from buffer into matrix
	  imageVec.vals[i * IMSIZE + j] = static_cast<float>( temp[i * IMSIZE + j] );
	}
      }
      fseek(file, k * IMSIZE * IMSIZE, SEEK_SET); // Seek to the kth image bytes

      Categorization label(10,dataid);

      dataItem x = {imageVec, label}; // Initialize an item with the data and the label in it
      _items.push_back(x); // Store in the vector

    }
    fclose(file);
  }
  return 0;
}


void Dataset::shuffle() {
  std::random_device r;
  std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()}; // Seed
  std::mt19937 eng1(seed); // Randomizer engine

  std::shuffle(begin(_items), end(_items), eng1); // Shuffle the dataset
  return; // todo add return codes instead of printing
}


std::vector<Dataset> Dataset::batch(int n) {
  std::vector<Dataset> batches;
  int noItems = this->_items.size();
  int noBatches = 0;
  std::vector<int> lines;

  std::size_t const half_size = lines.size() / 2;
  std::vector<int> split_lo(lines.begin(), lines.begin() + half_size);
  std::vector<int> split_hi(lines.begin() + half_size, lines.end());

  Dataset batchN;

  while (noItems >= n) {
    batchN._items = std::vector<dataItem>(
					  this->_items.begin() + noBatches * n, this->_items.begin() + noBatches * n + n);
    noBatches++;
    noItems -= n;
    batches.push_back(batchN);
  }

  if (noItems) { // If there are any remaining items, batch them up also
    batchN._items = std::vector<dataItem>(this->_items.begin() + noBatches * n, this->_items.end());
    noBatches++;
    batches.push_back(batchN);
  }

  return batches;
}

void Dataset::stack() { // Stacks vectors horizontally (column-wise) in a Matrix object
  //dataBatch  = VectorBatch( data_size(),  size(), 0);
  //labelBatch = VectorBatch( label_size(), size(), 0);
    
  dataBatch  = VectorBatch( size(),  data_size(), 0);
  labelBatch = VectorBatch( size(), label_size(), 0);

  for (int j = 0; j < size(); j++) {
    //dataBatch.set_col( j,data_vals(j) );
    dataBatch.set_row( j, data_vals(j) );
  }

  for (int j = 0; j < size(); j++) {
    //labelBatch.set_col( j,label_vals(j) );
    labelBatch.set_row( j, label_vals(j) );
  }
}


/*!
 * Split a dataset into a training and testing dataset
 */
std::pair< Dataset,Dataset > Dataset::split(float trainFraction) {
  int dataset_size = _items.size();
  int trainSize = ceil( static_cast<float>( dataset_size ) * trainFraction);
  int testSize = dataset_size - trainSize;

  Dataset trainSplit
    ( std::vector<dataItem>(this->_items.begin(), this->_items.begin() + trainSize) );

  Dataset testSplit
    ( std::vector<dataItem>(this->_items.begin() + trainSize, this->_items.end()) );

  return std::make_pair(trainSplit,testSplit);
}
