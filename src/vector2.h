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

#ifndef CODE_VEC2_H
#define CODE_VEC2_H

#include <vector>
#include "vector.h"
#include "matrix.h"
#include <iostream>
#ifdef BLISNN
#include "blis/blis.h"
#endif

#ifdef USE_GSL
#include "gsl/gsl-lite.hpp"
#endif

#define INDEXr(i,j,m,n) (i)*(n)+(j)
#define INDEXc(i,j,m,n) (i)+(j)*(m)

class VectorBatch{
  friend class Matrix;
  friend class Vector;

#if 0
  // hm. this doesn't work
  friend void relu_io    (const Vector &i, Vector &v);
  friend void sigmoid_io (const Vector &i, Vector &v);
  friend void softmax_io (const Vector &i, Vector &v);
  friend void none_io    (const Vector &i, Vector &v);
  friend void reluGrad_io(const Vector &m, Vector &a);
  friend void sigGrad_io (const Vector &m, Vector &a);

  friend void relu_io    (const VectorBatch &i, VectorBatch &v);
  friend void sigmoid_io (const VectorBatch &i, VectorBatch &v);
  friend void softmax_io (const VectorBatch &i, VectorBatch &v);
  friend void none_io    (const VectorBatch &i, VectorBatch &v);
  friend void reluGrad_io(const VectorBatch &m, VectorBatch &a);
  friend void sigGrad_io (const VectorBatch &m, VectorBatch &a);
#endif

private: //private:
  std::vector<float> vals;
  int nvectors{0},vector_size{0};
public:
    VectorBatch();
    VectorBatch( int itemsize );
    // this one is in the blis/reference file
    VectorBatch(int nRows, int nCols, bool rand=false);
    void allocate(int,int);

    int size() const { return vals.size(); };
    void resize(int m,int n) {
      nvectors = m; set_item_size(n); //r = m; c = n;
      vals.resize(m*n); };
    bool notnan() const {
      return all_of
	( vals.begin(),vals.end(),
	  [] (float e) { return not isnan(e); }
	);
    }
    bool notinf() const {
      return all_of
	( vals.begin(),vals.end(),
	  [] (float e) { return not isinf(e); }
	);
    }
    int item_size() const { return vector_size; };
    void set_item_size(int n) { vector_size = n; };
    int batch_size() const { return nvectors; };
    void set_batch_size(int n) { nvectors = n; };
    int nelements() const {
      int n = vals.size();
      assert( n==item_size()*batch_size() );
      return n;
    };
  //    VectorBatch transpose() const;
    std::vector<float>& vals_vector() { return vals; };
    const std::vector<float>& vals_vector() const { return vals; };
    float *data() { return vals.data(); };
    const float *data() const { return vals.data(); };
    const float *data( int disp ) const { return vals.data()+disp; };

	
	void v2mp( const Matrix &x, VectorBatch &y) const;
    void v2tmp( const Matrix &x, VectorBatch &y ) const;
	void v2mtp( const Matrix &x, VectorBatch &y ) const;
	void outer2( const VectorBatch &x, Matrix &y ) const;
	
  void add_vector( const std::vector<float> &v );
  void set_col(int j,const std::vector<float> &v );
  std::vector<float> get_col(int j) const;
  void set_row( int j, const std::vector<float> &v );
  std::vector<float> get_row(int j) const;
  std::vector<float> extract_vector(int v) const;
#ifdef USE_GSL
  gsl::span<float> get_vector(int v);
  //  const gsl::span<float> get_vector(int v) const;
#else
  std::vector<float> get_vector(int v) const;
#endif
  void set_vector( const Vector &v, int j);

  Vector get_vectorObj(int j) const {
    Vector vec(vector_size,0);
    for (int i = 0; i < vector_size; i++ ) 
      vec.vals.at(i) = vals.at( j * vector_size + i );
    return vec;
  }


  void show() const;
  void display(std::string) const;

  void addh(const Vector &y);
  void addh(const VectorBatch &y);
  Vector meanh() const;
	
  VectorBatch operator-(); // Unary negate operator
  VectorBatch& operator=(const VectorBatch& m2); // Copy constructor
  void hadamard(const VectorBatch& m1,const VectorBatch& m2);
  VectorBatch operator*(const VectorBatch &m2); // Hadamard Product Element-wise multiplication
  VectorBatch operator/(const VectorBatch &m2); // Element-wise division
  void scaleby( float );
  VectorBatch operator-(const VectorBatch &m2); // Element-wise subtraction
  friend VectorBatch operator/(const VectorBatch &m, const float &c); // for matrix-constant division
  friend VectorBatch operator*(const float &c, const VectorBatch &m); // for matrix-constant division
};


#endif
