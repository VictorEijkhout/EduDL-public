#include "vector2.h"
#include <iostream>
#include <vector>
#include <algorithm>

using std::vector;

/*
 * These are the vector bundle routines that do not have a optimized implementation,
 * such as using BLIS.
 */

VectorBundle::VectorBundle() { // Default constructor
  r = 0;
  c = 0;
  vals.clear();
}

void VectorBundle::addvh(const Vector &y) {
  for (int j = 0; j < c; j++) {
    for (int i = 0; i < y.size(); i++) {
      vals[j + c * i] += y.vals[i];
    }
  }
}

Vector VectorBundle::meanv() { // Returns a vector of column-wise means
  Vector mean(r, 0);
  float avg;
  for (int i = 0; i < r; i++) {
    avg = 0.0;
    for (int j = 0; j < c; j++) {
      avg += vals[i * c + j];
    }
    mean.vals[i] = avg;
  }
  return mean;
}


VectorBundle &VectorBundle::operator=(const VectorBundle &m2) { // Overloading the = operator
  r = m2.r;
  c = m2.c;

  this->vals = m2.vals; // IM Since we're using vectors we can just use the assignment from that

  return *this;
}


VectorBundle VectorBundle::operator-(const VectorBundle &m2) {
  VectorBundle out(m2.r, m2.c, 0);
  for (int i = 0; i < m2.r * m2.c; i++) {
    out.vals[i] = this->vals[i] - m2.vals[i];
  }
  return out;
}





VectorBundle VectorBundle::operator*(const VectorBundle &m2) { // Hadamard product
  VectorBundle out(m2.r, m2.c, 0);
  for (int i = 0; i < m2.r * m2.c; i++) {
    out.vals[i] = this->vals[i] * m2.vals[i];
  }
  return out;
}

VectorBundle VectorBundle::operator/(const VectorBundle &m2) { // Hadamard product
  VectorBundle out(m2.r, m2.c, 0);
  for (int i = 0; i < m2.r * m2.c; i++) {
    out.vals[i] = this->vals[i] / m2.vals[i];
  }
  return out;
}

VectorBundle VectorBundle::operator-() {
  VectorBundle result = *this;
  for (int i = 0; i < r * c; i++) {
    result.vals[i] = -vals[i];
  }

  return result;
};
