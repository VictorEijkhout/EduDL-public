#ifndef CODE_MAT_H
#define CODE_MAT_H

#include <vector>
#include "vector.h"
#include "vector2.h"
#include <initializer_list>

class Matrix{
public: // should really become private
  std::vector<float> mat;
  int r;
  int c;
public:
  Matrix();
  Matrix(int nRows, int nCols, int rand);
  // for mpl
  float *data();
  Matrix transpose() const;	
  const float *data() const { return mat.data(); };
  void show() const;
  //void flatten();
  void mvpt( const Vector &x, Vector &y ) const;
  void mvp( const Vector &x, Vector &y ) const;
  void addvh( const Vector &y); // Add a vector to each column
    
  void outerProduct( const Vector &x, const Vector &y );
  void outer2( const VectorBundle &x, const VectorBundle &y );
  Vector meanv();
  void zeros();
  void square();
	
  //vector2 methods
  void mv2p( const VectorBundle &x, VectorBundle &y) const;
  void mv2pt( const VectorBundle &x, VectorBundle &y ) const;
	

  Matrix operator-(); // Unary negate operator
  Matrix& operator=(const Matrix& m2); // Copy constructor
  Matrix operator+(const Matrix &m2) const; // Element-wise addition
  Matrix operator*(const Matrix &m2); // Hadamard Product Element-wise multiplication
  Matrix operator/(const Matrix &m2); // Element-wise division
  Matrix operator-(const Matrix &m2); // Element-wise subtraction
  friend Matrix operator*(const float &c, const Matrix &m); // for constant-matrix multiplication
  friend Matrix operator/(const Matrix &m, const float &c); // for matrix-constant division

};



#endif
