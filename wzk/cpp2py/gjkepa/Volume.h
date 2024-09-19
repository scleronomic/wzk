#ifndef _VOLUME_H_INCLUDED
#define _VOLUME_H_INCLUDED

#include <cmath>
#include <vector>
#include <limits>

#include "Vector.h"


//! A volume in space that is defined as the convex hull of n points plus a feather radius
/*! All links of the robots, objects in the surrounding as well as all intermediate volumes
    computed by the algorithm are described as the convex hull of a finite set of points plus
    a radius applied in all directions.

    This representation includes convex polygons (0 radius), spheres
    (1 point plus radius) and capped cylinders (2 points plus radius).

    Formally the volume contains all points which have a distance \c <=radius to a point
    \c sum_i lambda[i]*(*this)[i] with \c lambda[i]>=0 and \c sum_i lambda[i]==1.
*/



class Volume: public std::vector<Vector>
{
public:
    //! Empty volume
    Volume ():
    vector<Vector> (), radius (0)
    {}

    explicit Volume (std::vector<Vector>& v, float radius=0):
    vector<Vector> (v), radius (radius)
    {}

    Volume (Vector* v, int n, float radius=0):
    vector<Vector> (), radius (radius)
    {
      for (int i=0; i<n; i++)
          this->push_back(v[i]);
    }

    //! Additional radius added
    float radius;

    //! Returns the largest Euclidean norm of any vector inside the volume
    float maxNorm () const
    {
      float n2Max=0;
      for (int i=0; i<(int) size(); i++) {
          float norm2 = (*this)[i].sqrTwoNorm();
          if (norm2>n2Max) n2Max = norm2;
      }
      float result = sqrt(n2Max) + radius;
      assert (isfinite(result));
      return result;
    }

    float maxDot (int& argMax, const Vector& dir) const
    {
        float dotMax = -std::numeric_limits<float>::infinity();
        for (int i=0; i<size(); i++){
            float dot = dir.dot((*this)[i]);
            if (dot > dotMax) {
                dotMax = dot;
                argMax = i;
            }
        }
        return dotMax;
    }

    //! Computes that point of the volume that extends least into \c dir
    /*! Returns the minimal scalar product between \c dir and any point of the volume in \c minScp
      and the corresponding point in \c supportPoint.
      The computations includes \c radius.
    */
    //! Overloaded function returning the index of the supportPoint instead of the point itself
    void support (int& idx, float& length, const Vector& dir) const
    {
    assert (!empty());
        length = maxDot(idx, dir);
    }

  //! Returns whether this is a valid volume
  /*! There must be at least one point and all points as well as the radius must
      be finite and the radius >=0.
  */

    bool isValid () const
    {
      if (radius < 0) return false;
      if (empty()) return false;
      for (int i=0; i<(int) size(); i++)
          if (!isfinite((*this)[i].twoNorm()))
              return false;
      return true;
    }
};



#endif
