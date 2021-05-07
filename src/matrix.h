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

#ifndef CODE_MAT_H
#define CODE_MAT_H

#include <vector>
#include "vector.h"
//#include "vector2.h"
#include <initializer_list>

class Matrix{
private: // should really become private
	std::vector<float> mat;
    int r;
    int c;
public:
    Matrix();
    Matrix(int nRows, int nCols, int rand);
    // for mpl
    std::vector<float> &values() { return mat; };
    const std::vector<float> &values() const { return mat; };
    float* data() ;
    const float* data() const;
    int nelements() const {
      return mat.size();
    };
    int rowsize() const { return r; };
    int colsize() const { return c; };
    Matrix transpose() const;	
    void show() const;
    //void flatten();
    void mvpt( const Vector &x, Vector &y ) const;
    void mvp( const Vector &x, Vector &y ) const;
    void addvh( const Vector &y); // Add a vector to each column
    
	void mmp( const Matrix &x, Matrix &y) const;
    void outerProduct( const Vector &x, const Vector &y );
	Vector meanv();
    void zeros();
    void square();
    float normf() const;
	
	//vector2 methods
	// Moving these to VectorBatch object
	//void mv2p( const VectorBatch &x, VectorBatch &y) const;
    //void mv2pt( const VectorBatch &x, VectorBatch &y ) const;
    //void outer2( const VectorBatch &x, const VectorBatch &y );
	

	Matrix operator-(); // Unary negate operator
    Matrix& operator=(const Matrix& m2); // Copy constructor
    Matrix operator+(const Matrix &m2) const; // Element-wise addition
    Matrix operator*(const Matrix &m2); // Hadamard Product Element-wise multiplication
    Matrix operator/(const Matrix &m2); // Element-wise division
    Matrix operator-(const Matrix &m2); // Element-wise subtraction
    friend Matrix operator*(const float &c, const Matrix &m); // for constant-matrix multiplication
    friend Matrix operator/(const Matrix &m, const float &c); // for matrix-constant division
  void axpy( float a,const Matrix &x );

};



#endif
