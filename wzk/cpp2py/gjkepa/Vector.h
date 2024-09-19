//    This file contains very small self contained 3D vector class.
//    This allows to make  independent from third party libraries.

#ifndef VECTOR_INCLUDED
#define VECTOR_INCLUDED

#include <cmath>
#include <iostream>
#include <cstdlib>

//! A 3D vector
class Vector{
public:
    //! Initialize to zero vector
    Vector () {
          data[0]=data[1]=data[2]=0;
      }

    //! Copy constructor
    Vector (const Vector& v2) {
          *this = v2;
      }

    //! Initialize vector
    explicit Vector (const float myData[3]) {
          data[0]=myData[0];
          data[1]=myData[1];
          data[2]=myData[2];
      }

    //! Initialize vector with components
    Vector (float x, float y, float z) {
          data[0]=x;
          data[1]=y;
          data[2]=z;
      }

    //! Element access
    const float& operator[] (int i) const {
        return data[i];
    }

    //! Element access
    float& operator[] (int i) {
        return data[i];
    }

    //! Euclidean length
    float twoNorm () const {
        return sqrt(data[0]*data[0] + data[1]*data[1] + data[2]*data[2]);
    }

    //! Euclidean length squared, i.e. dot product with itself
    float sqrTwoNorm () const {
        return data[0]*data[0] + data[1]*data[1] + data[2]*data[2];
    }

    //! Make unit length
    Vector& normalize () {
        *this *= 1/twoNorm();
        return *this;
    }

    //! Negate vector
    Vector operator - () const {
        return Vector(-data[0], -data[1], -data[2]);}

    //! Add to the vector
    Vector& operator += (const Vector& v2) {
        data[0] += v2.data[0];
        data[1] += v2.data[1];
        data[2] += v2.data[2];
        return *this;
    }

    //! Subtract from the vector
    Vector& operator -= (const Vector& v2) {
        data[0] -= v2.data[0];
        data[1] -= v2.data[1];
        data[2] -= v2.data[2];
        return *this;
    }

    //! Sum of two vectors
    Vector operator + (const Vector& v2) const {
        return Vector (data[0]+v2.data[0], data[1]+v2.data[1], data[2]+v2.data[2]);
    }

    //! Difference of two vectors
    Vector operator - (const Vector& v2) const {
        return Vector (data[0]-v2.data[0], data[1]-v2.data[1], data[2]-v2.data[2]);
    }

    //! Dot product
    float dot (const Vector& v2) const {
        return data[0]*v2.data[0] + data[1]*v2.data[1] + data[2]*v2.data[2];
    }

    //! Cross product
    Vector cross (const Vector& v2) const {
    return Vector (data[1]*v2.data[2] - data[2]*v2.data[1],
                   data[2]*v2.data[0] - data[0]*v2.data[2],
                   data[0]*v2.data[1] - data[1]*v2.data[0]);
    }

    //! Times scalar
    Vector operator * (float lambda) const {
        return Vector (lambda*data[0], lambda*data[1], lambda*data[2]);}

    //! Times scalar
    Vector& operator *= (float lambda) {
        data[0]*=lambda; data[1]*=lambda; data[2]*=lambda; return * this;}

    //! exact equality
    bool operator == (const Vector& v2) {
        return data[0]==v2.data[0] && data[1]==v2.data[1] && data[2]==v2.data[2];}

    friend Vector operator * (float lambda, const Vector& v);


protected:
    //! The actual 3 components
    float data[3]{};
};

inline Vector normalize(const Vector& v){
    return v * (1/v.twoNorm());
}

inline Vector cross(const Vector& v1, const Vector& v2){
    return v1.cross(v2);
}

inline float dot(const Vector& v1, const Vector& v2){
    return v1.dot(v2);
}

//! Times scalar
inline Vector operator * (float lambda, const Vector& v) {
    return Vector (lambda*v.data[0], lambda*v.data[1], lambda*v.data[2]);}
//! Print vector
inline std::ostream& operator << (std::ostream& f, const Vector& v) { f << "[ "<<v[0]<<", "<<v[1]<<", "<<v[2]<<" ]"; return f; }

#endif
