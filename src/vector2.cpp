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

#include "vector2.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
using std::vector;

/*
 * These are the vector batch routines that do not have a optimized implementation,
 * such as using BLIS.
 */

VectorBatch::VectorBatch() { // Default constructor
    r = 0;
    c = 0;
    vals.clear();
}

void VectorBatch::addh(const Vector &y) { // Add y to every row
	assert( c == y.size() );
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++ ) {
			vals[ i*c + j ] += y.vals[j];
		}
	}
}

Vector VectorBatch::meanh() { // Returns a vector of row-wise means
    Vector mean(c, 0);
    float avg;
	for (int i = 0; i < c; i++) {
		avg = 0.0;
		for ( int j = 0; j < r; j++ ) {
			avg += vals[ j * c + i ];
		}
		mean.vals[i] = avg/r;
	}
    return mean;
}


VectorBatch &VectorBatch::operator=(const VectorBatch &m2) { // Overloading the = operator
    r = m2.r;
    c = m2.c;

    this->vals = m2.vals; // IM Since we're using vectors we can just use the assignment from that

    return *this;
}


VectorBatch VectorBatch::operator-(const VectorBatch &m2) {
    VectorBatch out(m2.r, m2.c, 0);
    for (int i = 0; i < m2.r * m2.c; i++) {
        out.vals[i] = this->vals[i] - m2.vals[i];
    }
    return out;
}

VectorBatch VectorBatch::operator*(const VectorBatch &m2) { // Hadamard product
    VectorBatch out(m2.r, m2.c, 0);
    for (int i = 0; i < m2.r * m2.c; i++) {
        out.vals[i] = this->vals[i] * m2.vals[i];
    }
    return out;
}

VectorBatch VectorBatch::operator/(const VectorBatch &m2) { // Hadamard product
    VectorBatch out(m2.r, m2.c, 0);
    for (int i = 0; i < m2.r * m2.c; i++) {
        out.vals[i] = this->vals[i] / m2.vals[i];
    }
    return out;
}

VectorBatch VectorBatch::operator-() {
    VectorBatch result = *this;
    for (int i = 0; i < r * c; i++) {
        result.vals[i] = -vals[i];
    }

    return result;
};

VectorBatch operator/(const VectorBatch &m, const float &c) {
    VectorBatch o = m;
    for (int i = 0; i < o.r * o.c; i++) {
        o.vals[i] = o.vals[i] / c;
    }
    return o;
}

VectorBatch operator*(const float &c, const VectorBatch &m) {
    VectorBatch o = m;
    for (int i = 0; i < o.r * o.c; i++) {
        o.vals[i] = o.vals[i] * c;
    }
    return o;
}
