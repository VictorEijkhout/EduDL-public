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
using std::cout;
using std::endl;
#include <string>
using std::string;
#include <vector>
using std::vector;

#include <cstdio>
#include <cassert>
#include <algorithm>
#include <random>

#define IMSIZE 28

Dataset::Dataset( int n ) : nclasses(n) {
  assert(n>=0);
};

Dataset::Dataset( std::vector<dataItem> dv ) {
  for ( const auto& v : dv )
    push_back(v);
};

void Dataset::set_lowerbound( int b ) { lowerbound = b; };
void Dataset::set_number( int b ) { number = b; };

/*!
 * Add a new data item, and check its consistency with previous items
 */
void Dataset::push_back(dataItem it) {
  if (nclasses>0 and it.label_size()!=nclasses) {
    cout << "Set dimensionality " << nclasses
         << " does not match item size " << it.label_size() << endl;
    throw( string("Fail to add item to dataset") );
  }
  if (nclasses==0)
    nclasses = it.label_size();
  dataBatch.add_vector( it.data_values() );
  labelBatch.add_vector( it.label_values() );
  //_items.push_back(it);
};

dataItem Dataset::item(int i) const {
  return dataItem( data_vals(i),label_vals(i) );
};

int Dataset::size() const {
  int ds = dataBatch.batch_size(), ls = labelBatch.batch_size();
  assert( ds==ls );
  return ds;
}

/*!
 * What is the size of the feature vector in this dataset?
 */
int Dataset::data_size() const {
  if (dataBatch.batch_size()==0)
    throw( string("Can not get data size for empty dataset") );
  return dataBatch.item_size();
};

/*!
 * What is the number of categories in the labels of this dataset?
 */
int Dataset::label_size() const {
  if (labelBatch.batch_size()==0)
    throw( string("Can not get label size for empty dataset") );
  return labelBatch.item_size();
}

/*!
 * Get the i-th data object 
 */
const Vector& Dataset::data(int i) const {
  throw( string("Do not use Dataset::data") );
  //  return _items.at(i).data;
};
/*!
 * Get the features of i-th data object 
 */
const vector<float> Dataset::data_vals(int i) const {
  return dataBatch.extract_vector(i);
};
/*!
 * Get the categorization of i-th data object 
 */
const vector<float> Dataset::label_vals(int i) const {
  return labelBatch.extract_vector(i);
};
//! Same, of the stacked object
vector<float> Dataset::stacked_data_vals(int i) const {
  throw( string("Do not use Dataset::stacked_data_vals") );
  return dataBatch.get_row(i);
};

//! Same, of the stacked object
vector<float> Dataset::stacked_label_vals(int i) const {
  throw( string("Do not use Dataset::stacked_label_vals") );
  return labelBatch.get_row(i);
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
            cout << "Error opening file" << endl;
            return -2; // Arbitrary error code
        } 
        for (int k = 0; k < 1000; k++) {
            Vector imageVec(IMSIZE * IMSIZE, 0); // initialize matrix to be read into
            fread(temp, 1, IMSIZE * IMSIZE, file); // Read 28*28 into buffer

            for (int i = 0; i < IMSIZE; i++) {
                for (int j = 0; j < IMSIZE; j++) {
                    // Transfer from buffer into matrix
		  float *i_data = imageVec.data();
		  *( i_data + i * IMSIZE + j ) // imageVec.vals[i * IMSIZE + j]
		    = static_cast<float>( temp[i * IMSIZE + j] );
                }
            }
            fseek(file, k * IMSIZE * IMSIZE, SEEK_SET); // Seek to the kth image bytes

	    Categorization label(10,dataid);

            dataItem x = {imageVec, label}; // Initialize an item with the data and the label in it
            push_back(x); // Store in the vector

        }
        fclose(file);
    }
    return 0;
}


void Dataset::shuffle() {
    std::random_device r;
    std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()}; // Seed
    std::mt19937 eng1(seed); // Randomizer engine

    throw( string("shuffling doesn't work") );
    //    std::shuffle(begin(_items), end(_items), eng1); // Shuffle the dataset
    return; // todo add return codes instead of printing
}


std::vector<Dataset> Dataset::batch(int batch_size) const {

  std::vector<Dataset> batches;
  int nitems = size(), itemsize = data_size();
  int nbatches = nitems/batch_size + ( nitems%batch_size>0 ? 1 : 0 );
  for (int b=0; b<nbatches; b++) {
    int first = b*batch_size, last= std::min( (b+1)*batch_size,nitems );
    Dataset batch; //(itemsize,last-first);
    batch.set_lowerbound(first); batch.set_number(b);
    for (int i=first; i<last; i++) {
      batch.push_back( item(i) );
    }
    batches.push_back(batch);
  }

  return batches;
}

void Dataset::stack() { // Stacks vectors horizontally (column-wise) in a Matrix object
  throw( string("Stacking no longer needed") );
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
std::pair< Dataset,Dataset > Dataset::split(float trainFraction) const {
    int dataset_size = size(); // _items.size();
    int testSize{0},trainSize;
    while ( true ) {
      trainSize= ceil( static_cast<float>( dataset_size ) * trainFraction);
      testSize = dataset_size - trainSize;
      if ( testSize>0 ) break;
      trainFraction *= .9;
    }

    Dataset trainSplit;    
#ifdef DEBUG
    cout << "split into " << trainSize << "+" << testSize << endl;
#endif
    for (int i=0; i<trainSize; i++) {
      const auto& di = item(i);
      trainSplit.push_back(di);
    }
    Dataset testSplit;
    for (int i=trainSize; i<trainSize+testSize; i++) {
      const auto& di = item(i);
      testSplit.push_back(di);
    }

    // Dataset trainSplit
    //   ( std::vector<dataItem>(this->_items.begin(), this->_items.begin() + trainSize) );

    // Dataset testSplit
    //   ( std::vector<dataItem>(this->_items.begin() + trainSize, this->_items.end()) );

    return std::make_pair(trainSplit,testSplit);
}
