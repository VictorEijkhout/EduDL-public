#ifndef CODE_VEC2_H
#define CODE_VEC2_H

#include <vector>
#include "vector.h"

#ifdef BLISNN
#include "blis/blis.h"
#endif

class VectorBundle{
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

  friend void relu_io    (const VectorBundle &i, VectorBundle &v);
  friend void sigmoid_io (const VectorBundle &i, VectorBundle &v);
  friend void softmax_io (const VectorBundle &i, VectorBundle &v);
  friend void none_io    (const VectorBundle &i, VectorBundle &v);
  friend void reluGrad_io(const VectorBundle &m, VectorBundle &a);
  friend void sigGrad_io (const VectorBundle &m, VectorBundle &a);
#endif

public: //private:
  std::vector<float> vals;
public:
    int r;
    int c;
    VectorBundle();
    VectorBundle(int nRows, int nCols, bool rand=false);

    int size() const { return vals.size(); };
    int item_size() const { return r; };
	int bundle_size() const { return c; };
    VectorBundle transpose() const;
    std::vector<float>& vals_vector() { return vals; };
    const std::vector<float>& vals_vector() const { return vals; };
    float *data() { return vals.data(); };
    const float *data() const { return vals.data(); };
    void set_col(int j,const std::vector<float> &v ) {
      assert( v.size()==r );
      for (int i = 0; i < r; i++) {
	vals.at( j + c * i ) = v.at(i);
      };
    };
    std::vector<float> get_col(int j) const {
      std::vector<float> col(r);
      for (int i=0; i<r; i++)
	col.at(i) = vals.at( j + c*i );
      return col;
    };
    std::vector<float> get_vector(int v) const {
      return get_col(v);
    };
    
	Vector get_vectorObj(int j) const {
	  Vector vec(r,0);
	  for (int i=0; i<r; i++)
	    vec.vals.at(i) = vals.at( j+c*i );
	return vec;
	}

	void set_vector( const Vector &v, int j) {
	  for (int i=0; i<r; i++)
	    vals.at( j + c*i ) = v.vals.at(i); 
	}
	void show() const;

	void addvh(const Vector &y);
	Vector meanv();
	
	VectorBundle operator-(); // Unary negate operator
    VectorBundle& operator=(const VectorBundle& m2); // Copy constructor
    VectorBundle operator*(const VectorBundle &m2); // Hadamard Product Element-wise multiplication
    VectorBundle operator/(const VectorBundle &m2); // Element-wise division
    VectorBundle operator-(const VectorBundle &m2); // Element-wise subtraction
	friend VectorBundle operator/(const VectorBundle &m, const float &c); // for matrix-constant division
	friend VectorBundle operator*(const float &c, const VectorBundle &m); // for matrix-constant division
};


#endif
