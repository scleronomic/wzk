#ifndef _TETRAHEDRON_H_INCLUDED
#define _TETRAHEDRON_H_INCLUDED

#include <vector>
#include <cassert>
#include "Vector.h"

//! One corner of the tetrahedron
/*! Both the actual point and all indices needed to backtrack the original object's
corners. */
class Corner{
public:
    //! Which object vertices formed this corner
    /*! The corner is the difference of vertex \c aIdx form object \c a and \c bIdx
        from object \c b. If this corner is not actually existing it is -1.
    */
    int aIdx, bIdx;

    //! The actual coordinates (i.e. the difference in a common frame)
    Vector v;

    //! The coefficient in the convex combination of the point closest to 0.
    /*! If the closest point lies on a face, edge, corner, i.e. lambda would
        be 0 in real arithmetic
        it is also guaranteed to be \c ==0 numerically.
    */
    float lambda;

    //! Empty corner
    Corner ():
            aIdx(-1), bIdx(-1), v(), lambda(0)
    {}

    //! Standard constructor
    Corner (int aIdx, int bIdx, const Vector& v):
            aIdx(aIdx), bIdx(bIdx), v(v), lambda(0)
    {}

    //! Whether this corner is existing
    /*! Objects with less than 4 are represented by setting \c aIdx and \c bIdx
        to 0 in the remaining corner objects.
    */
    bool isValid () const {
        return aIdx>=0;
    }

    //! Flags the corner as not existing
    void clear () {
        aIdx=bIdx=-1;
        v[0]=v[1]=v[2]=0;
        lambda=0;
    }
};


//! The convex hull of four points or less in the Minkowski difference of two \c Volume
/*! This is a specialized class for representing the intermediate tetrahedrons (or lower
    dimensional simplices) in the Minkowski difference of two \c Volume objects. These
    tetrahedrons represent the internal state of the GJK algorithm and a central routine of
    the inner loop is to compute the point on that tetrahedron being closest to the origin.
    This class not only contains the up to four corners but also two object-vertex indices
    for each corner that allow to identify the original objects vertices that contributed
    to this tetrahedron in the Minkowski difference.

    So this class has two purposes:
    a) Providing an interface for \c closestPoint() and
    b) storing the intermediate state of the GJK algorithm for an object pair during
       iteration and between successive calls.
 */
class Simplex{
public:

    //! Up to four corners
    //! /*! If there are less than four the last ones must be \c !valid(). */
    Corner corner[4];

    //! Number of corners in \c corner
    /*! Always less equal 4 */
    int nrCorners;

    //! Empty tetrahedron
    Simplex () {
        corner[0]=corner[1]=corner[2]=corner[3]=Corner();
        nrCorners=0;
    }

    void removeAll0Lambdas()
    {
        switch (nrCorners)
        {
            case 1:
                removeAll0Lambdas1();
                break;
            case 2:
                removeAll0Lambdas2();
                break;
            case 3:
                removeAll0Lambdas3();
                break;
            case 4:
                removeAll0Lambdas4();
                break;
        }
    }

    void removeAll0Lambdas1 ()
    {
        if (corner[0].lambda>0)
            nrCorners = 1;
        else {
            corner[0].clear();
            nrCorners = 0;
    }
  }

    void removeAll0Lambdas2 ()
    {
        if (corner[0].lambda>0) {
            if (corner[1].lambda>0)
                nrCorners = 2;
            else {
                corner[1].clear();
                nrCorners = 1;
            }
        }
        else {
            if (corner[1].lambda>0) {
                corner[0] = corner[1];
                corner[1].clear();
                nrCorners = 1;
            }
            else {
                corner[0].clear();
                corner[1].clear();
                nrCorners = 0;
            }
        }
    }

    void removeAll0Lambdas3 ()
    {
        bool l0 = corner[0].lambda>0;
        bool l1 = corner[1].lambda>0;
        bool l2 = corner[2].lambda>0;
        if (!l0){
            if (l2)
            {
                corner[0] = corner[2];
                corner[2].clear();
                if (l1)
                    nrCorners=2;
                else {
                    corner[1].clear();
                    nrCorners = 1;
                }
            }
            else {
        if (l1) {
          corner[0] = corner[1];
          nrCorners = 1;
        }
        else {
          nrCorners = 0; 
          corner[0].clear();
        }
        corner[1].clear();
        corner[2].clear();
      }
        }
        else
        {
            if (!l1) {
                if (l2) {
                    corner[1] = corner[2];
                    corner[2].clear();
                    nrCorners = 2;
            }
                else {
                    corner[1].clear();
                    corner[2].clear();
                    nrCorners = 1;
            }
            }
            else {
                if (l2)
                    nrCorners = 3;
                else {
                    corner[2].clear();
                    nrCorners = 2;
                }
            }
        }
    }

    void removeAll0Lambdas4 ()
    {
        nrCorners = 0;
        for (int j=0; j<4; j++) {
            if (corner[j].lambda>0) {
                if (nrCorners<j)
                    corner[nrCorners] = corner[j];
                nrCorners++;
            }
        }
        for (int i=nrCorners; i<4; i++)
            corner[i].clear();
    }


    void closestPoint()
    {
        switch (nrCorners)
        {
            case 1:
                closestPoint1();
                break;
            case 2:
                closestPoint2();
                break;
            case 3:
                closestPoint3();
                break;
            case 4:
                closestPoint4();
                break;
        }
    }

    void closestPoint1 ()
    {
        corner[0].lambda = 1;
    }

    void closestPoint2 ()
    {
        // 1. Compute all scalar products needed
        float s00 = corner[0].v.dot(corner[0].v);
        float s01 = corner[0].v.dot(corner[1].v);
        float s11 = corner[1].v.dot(corner[1].v);

        // 2. Compute the delta coefficients
        float d0_01 = (s11-s01);
        float d1_01 = (s00-s01);
        float d_01 = d0_01 + d1_01;

        // 3. Check using the d*'s which convex subspace contains the closest point
        if (d1_01<=0) {
            corner[0].lambda=1;
            corner[1].lambda=0;
            return;
        }
        if (d0_01<=0) {
            corner[0].lambda=0;
            corner[1].lambda=1;
            return;
        }
        if (d0_01>0 && d1_01>0) {
            float sumInv = 1/d_01;
            corner[0].lambda=d0_01*sumInv;
            corner[1].lambda=d1_01*sumInv;
            return;
        }

        // 4. Backup code: If we get here we have encountered a numerical problem
        // because in theory one of the above cases should apply
        // So we choose that subspace that yields the smallest distance
        float d2Min = s00;
        int d2MinIdx = 1;
        float d2 = (s00);
        if (d2 < d2Min) {
            d2Min = d2;
            d2MinIdx = 1;
        }

        d2 = (s11);
        if (d2<d2Min) {
            d2Min=d2;
            d2MinIdx=2;
        }

        if (d0_01>0 && d1_01>0) {
            d2 = (d0_01*s00+d1_01*s01)/d_01;
            if (d2<d2Min) {
                d2MinIdx=3;
            }
        }

        if (d2MinIdx==1) {
            corner[0].lambda=1;
            corner[1].lambda=0;
            return;
        }
        if (d2MinIdx==2) {
            corner[0].lambda=0;
            corner[1].lambda=1;
            return;
        }
        if (d2MinIdx==3) {
            float sumInv = 1/d_01;
            corner[0].lambda=d0_01*sumInv;
            corner[1].lambda=d1_01*sumInv;
            return;
        }
    }

    void closestPoint3 ()
    {
        // 1. Compute all scalar products needed
        float s00 = corner[0].v.dot(corner[0].v);
        float s01 = corner[0].v.dot(corner[1].v);
        float s02 = corner[0].v.dot(corner[2].v);
        float s11 = corner[1].v.dot(corner[1].v);
        float s12 = corner[1].v.dot(corner[2].v);
        float s22 = corner[2].v.dot(corner[2].v);

        // 2. Compute the delta coefficients
        float d0_01 = (s11-s01);
        float d1_01 = (s00-s01);
        float d_01 = d0_01 + d1_01;
        float d0_02 = (s22-s02);
        float d2_02 = (s00-s02);
        float d_02 = d0_02 + d2_02;
        float d1_12 = (s22-s12);
        float d2_12 = (s11-s12);
        float d_12 = d1_12 + d2_12;
        float d0_012 = d1_12*(s11-s01) + d2_12*(s12-s02);
        float d1_012 = d0_02*(s00-s01) + d2_02*(s02-s12);
        float d2_012 = d0_01*(s00-s02) + d1_01*(s01-s12);
        float d_012 = d0_012 + d1_012 + d2_012;

        // 3. Check using the d*'s which convex subspace contains the closest point
        if (d1_01<=0 && d2_02<=0) {
            corner[0].lambda=1;
            corner[1].lambda=0;
            corner[2].lambda=0;
            return;
        }
        if (d0_01<=0 && d2_12<=0) {
            corner[0].lambda=0;
            corner[1].lambda=1;
            corner[2].lambda=0;
            return;
        }
        if (d0_02<=0 && d1_12<=0) {
            corner[0].lambda=0;
            corner[1].lambda=0;
            corner[2].lambda=1;
            return;
        }
        if (d0_01>0 && d1_01>0 && d2_012<=0) {
            float sumInv = 1/d_01;
            corner[0].lambda=d0_01*sumInv;
            corner[1].lambda=d1_01*sumInv;
            corner[2].lambda=0;
            return;
        }
        if (d0_02>0 && d1_012<=0 && d2_02>0) {
            float sumInv = 1/d_02;
            corner[0].lambda=d0_02*sumInv;
            corner[1].lambda=0;
            corner[2].lambda=d2_02*sumInv;
            return;
        }
        if (d0_012<=0 && d1_12>0 && d2_12>0) {
            float sumInv = 1/d_12;
            corner[0].lambda=0;
            corner[1].lambda=d1_12*sumInv;
            corner[2].lambda=d2_12*sumInv;
            return;
        }
        if (d0_012>0 && d1_012>0 && d2_012>0) {
            float sumInv = 1/d_012;
            corner[0].lambda=d0_012*sumInv;
            corner[1].lambda=d1_012*sumInv;
            corner[2].lambda=d2_012*sumInv;
            return;
        }

        // 4. Backup code: If we get here we have encountered a numerical problem
        // because in theory one of the above cases should apply
        // So we choose that subspace that yields the smallest distance
        float d2Min = s00;
        int d2MinIdx = 1;
        float d2 = (s00);
        if (d2 < d2Min) {
            d2Min = d2;
            d2MinIdx = 1;
        }
        d2 = (s11);
        if (d2 < d2Min) {
            d2Min = d2;
            d2MinIdx = 2;
        }
        if (d0_01>0 && d1_01>0) {
            d2 = (d0_01*s00+d1_01*s01)/d_01;
            if (d2<d2Min) {
                d2Min=d2;
                d2MinIdx=3;}
        }
        d2 = (s22);
        if (d2 < d2Min) {
            d2Min = d2;
            d2MinIdx = 4;
        }
        if (d0_02>0 && d2_02>0) {
            d2 = (d0_02*s00+d2_02*s02)/d_02;
            if (d2<d2Min) {
                d2Min=d2;
                d2MinIdx=5;}
        }
        if (d1_12>0 && d2_12>0) {
            d2 = (d1_12*s11+d2_12*s12)/d_12;
            if (d2<d2Min){
                d2Min=d2;
                d2MinIdx=6;
            }
        }
        if (d0_012>0 && d1_012>0 && d2_012>0) {
            d2 = (d0_012*s00+d1_012*s01+d2_012*s02)/d_012;
            if (d2<d2Min){
                d2MinIdx=7;
            }
        }
        if (d2MinIdx==1) {
            corner[0].lambda=1;
            corner[1].lambda=0;
            corner[2].lambda=0;
            return;
        }
        if (d2MinIdx==2) {
            corner[0].lambda=0;
            corner[1].lambda=1;
            corner[2].lambda=0;
            return;
        }
        if (d2MinIdx==3) {
            float sumInv = 1/d_01;
            corner[0].lambda=d0_01*sumInv;
            corner[1].lambda=d1_01*sumInv;
            corner[2].lambda=0;
            return;
        }
        if (d2MinIdx==4) {
            corner[0].lambda=0;
            corner[1].lambda=0;
            corner[2].lambda=1;
            return;
        }
        if (d2MinIdx==5) {
            float sumInv = 1/d_02;
            corner[0].lambda=d0_02*sumInv;
            corner[1].lambda=0;
            corner[2].lambda=d2_02*sumInv;
            return;
        }
        if (d2MinIdx==6) {
            float sumInv = 1/d_12;
            corner[0].lambda=0;
            corner[1].lambda=d1_12*sumInv;
            corner[2].lambda=d2_12*sumInv;
            return;
        }
        if (d2MinIdx==7) {
            float sumInv = 1/d_012;
            corner[0].lambda=d0_012*sumInv;
            corner[1].lambda=d1_012*sumInv;
            corner[2].lambda=d2_012*sumInv;
            return;
        }
    }

    void closestPoint4 ()
    {
        // 1. Compute all scalar products needed
        float s00 = corner[0].v.dot(corner[0].v);
        float s01 = corner[0].v.dot(corner[1].v);
        float s02 = corner[0].v.dot(corner[2].v);
        float s03 = corner[0].v.dot(corner[3].v);
        float s11 = corner[1].v.dot(corner[1].v);
        float s12 = corner[1].v.dot(corner[2].v);
        float s13 = corner[1].v.dot(corner[3].v);
        float s22 = corner[2].v.dot(corner[2].v);
        float s23 = corner[2].v.dot(corner[3].v);
        float s33 = corner[3].v.dot(corner[3].v);

        // 2. Compute the delta coefficients
        float d0_01 = (s11-s01);
        float d1_01 = (s00-s01);
        float d_01 = d0_01 + d1_01;
        float d0_02 = (s22-s02);
        float d2_02 = (s00-s02);
        float d_02 = d0_02 + d2_02;
        float d1_12 = (s22-s12);
        float d2_12 = (s11-s12);
        float d_12 = d1_12 + d2_12;
        float d0_03 = (s33-s03);
        float d3_03 = (s00-s03);
        float d_03 = d0_03 + d3_03;
        float d1_13 = (s33-s13);
        float d3_13 = (s11-s13);
        float d_13 = d1_13 + d3_13;
        float d2_23 = (s33-s23);
        float d3_23 = (s22-s23);
        float d_23 = d2_23 + d3_23;
        float d0_012 = d1_12*(s11-s01) + d2_12*(s12-s02);
        float d1_012 = d0_02*(s00-s01) + d2_02*(s02-s12);
        float d2_012 = d0_01*(s00-s02) + d1_01*(s01-s12);
        float d_012 = d0_012 + d1_012 + d2_012;
        float d0_013 = d1_13*(s11-s01) + d3_13*(s13-s03);
        float d1_013 = d0_03*(s00-s01) + d3_03*(s03-s13);
        float d3_013 = d0_01*(s00-s03) + d1_01*(s01-s13);
        float d_013 = d0_013 + d1_013 + d3_013;
        float d0_023 = d2_23*(s22-s02) + d3_23*(s23-s03);
        float d2_023 = d0_03*(s00-s02) + d3_03*(s03-s23);
        float d3_023 = d0_02*(s00-s03) + d2_02*(s02-s23);
        float d_023 = d0_023 + d2_023 + d3_023;
        float d1_123 = d2_23*(s22-s12) + d3_23*(s23-s13);
        float d2_123 = d1_13*(s11-s12) + d3_13*(s13-s23);
        float d3_123 = d1_12*(s11-s13) + d2_12*(s12-s23);
        float d_123 = d1_123 + d2_123 + d3_123;
        float d0_0123 = d1_123*(s11-s01) + d2_123*(s12-s02) + d3_123*(s13-s03);
        float d1_0123 = d0_023*(s00-s01) + d2_023*(s02-s12) + d3_023*(s03-s13);
        float d2_0123 = d0_013*(s00-s02) + d1_013*(s01-s12) + d3_013*(s03-s23);
        float d3_0123 = d0_012*(s00-s03) + d1_012*(s01-s13) + d2_012*(s02-s23);
        float d_0123 = d0_0123 + d1_0123 + d2_0123 + d3_0123;

        // 3. Check using the d*'s which convex subspace contains the closest point
        if (d1_01<=0 && d2_02<=0 && d3_03<=0) {
            corner[0].lambda=1;
            corner[1].lambda=0;
            corner[2].lambda=0;
            corner[3].lambda=0;
            return;
        }
        if (d0_01<=0 && d2_12<=0 && d3_13<=0) {
            corner[0].lambda=0;
            corner[1].lambda=1;
            corner[2].lambda=0;
            corner[3].lambda=0;
            return;
        }
        if (d0_02<=0 && d1_12<=0 && d3_23<=0) {
            corner[0].lambda=0;
            corner[1].lambda=0;
            corner[2].lambda=1;
            corner[3].lambda=0;
            return;
        }
        if (d0_03<=0 && d1_13<=0 && d2_23<=0) {
            corner[0].lambda=0;
            corner[1].lambda=0;
            corner[2].lambda=0;
            corner[3].lambda=1;
            return;
        }
        if (d0_01>0 && d1_01>0 && d2_012<=0 && d3_013<=0) {
            float sumInv = 1/d_01;
            corner[0].lambda=d0_01*sumInv;
            corner[1].lambda=d1_01*sumInv;
            corner[2].lambda=0;
            corner[3].lambda=0;
            return;
        }
        if (d0_02>0 && d1_012<=0 && d2_02>0 && d3_023<=0) {
            float sumInv = 1/d_02;
            corner[0].lambda=d0_02*sumInv;
            corner[1].lambda=0;
            corner[2].lambda=d2_02*sumInv;
            corner[3].lambda=0;
            return;
        }
        if (d0_012<=0 && d1_12>0 && d2_12>0 && d3_123<=0) {
            float sumInv = 1/d_12;
            corner[0].lambda=0;
            corner[1].lambda=d1_12*sumInv;
            corner[2].lambda=d2_12*sumInv;
            corner[3].lambda=0;
            return;
        }
        if (d0_03>0 && d1_013<=0 && d2_023<=0 && d3_03>0) {
            float sumInv = 1/d_03;
            corner[0].lambda=d0_03*sumInv;
            corner[1].lambda=0;
            corner[2].lambda=0;
            corner[3].lambda=d3_03*sumInv;
            return;
        }
        if (d0_013<=0 && d1_13>0 && d2_123<=0 && d3_13>0) {
            float sumInv = 1/d_13;
            corner[0].lambda=0;
            corner[1].lambda=d1_13*sumInv;
            corner[2].lambda=0;
            corner[3].lambda=d3_13*sumInv;
            return;
        }
        if (d0_023<=0 && d1_123<=0 && d2_23>0 && d3_23>0) {
            float sumInv = 1/d_23;
            corner[0].lambda=0;
            corner[1].lambda=0;
            corner[2].lambda=d2_23*sumInv;
            corner[3].lambda=d3_23*sumInv;
            return;
        }
        if (d0_012>0 && d1_012>0 && d2_012>0 && d3_0123<=0) {
            float sumInv = 1/d_012;
            corner[0].lambda=d0_012*sumInv;
            corner[1].lambda=d1_012*sumInv;
            corner[2].lambda=d2_012*sumInv;
            corner[3].lambda=0;
            return;
        }
        if (d0_013>0 && d1_013>0 && d2_0123<=0 && d3_013>0) {
            float sumInv = 1/d_013;
            corner[0].lambda=d0_013*sumInv;
            corner[1].lambda=d1_013*sumInv;
            corner[2].lambda=0;
            corner[3].lambda=d3_013*sumInv;
            return;
        }
        if (d0_023>0 && d1_0123<=0 && d2_023>0 && d3_023>0) {
            float sumInv = 1/d_023;
            corner[0].lambda=d0_023*sumInv;
            corner[1].lambda=0;
            corner[2].lambda=d2_023*sumInv;
            corner[3].lambda=d3_023*sumInv;
            return;
        }
        if (d0_0123<=0 && d1_123>0 && d2_123>0 && d3_123>0) {
            float sumInv = 1/d_123;
            corner[0].lambda=0;
            corner[1].lambda=d1_123*sumInv;
            corner[2].lambda=d2_123*sumInv;
            corner[3].lambda=d3_123*sumInv;
            return;
        }
        if (d0_0123>0 && d1_0123>0 && d2_0123>0 && d3_0123>0) {
            float sumInv = 1/d_0123;
            corner[0].lambda=d0_0123*sumInv;
            corner[1].lambda=d1_0123*sumInv;
            corner[2].lambda=d2_0123*sumInv;
            corner[3].lambda=d3_0123*sumInv;
            return;
        }

        // 4. Backup code: If we get here we have encountered a numerical problem
        // because in theory one of the above cases should apply
        // So we choose that subspace that yields the smallest distance
        float d2Min = s00;
        int d2MinIdx = 1;
        float d2 = (s00);
        if (d2 < d2Min) {
            d2Min = d2;
            d2MinIdx = 1;
        }
        d2 = (s11);
        if (d2 < d2Min) {
            d2Min = d2;
            d2MinIdx = 2;
        }
        if (d0_01>0 && d1_01>0) {
            d2 = (d0_01*s00+d1_01*s01)/d_01;
            if (d2<d2Min) {
                d2Min=d2;
                d2MinIdx=3;
            }
        }
        d2 = (s22);
        if (d2 < d2Min) {
            d2Min = d2;
            d2MinIdx = 4;
        }
        if (d0_02>0 && d2_02>0) {
            d2 = (d0_02*s00+d2_02*s02)/d_02;
            if (d2<d2Min) {d2Min=d2;d2MinIdx=5;}
        }
        if (d1_12>0 && d2_12>0) {
            d2 = (d1_12*s11+d2_12*s12)/d_12;
            if (d2<d2Min) {d2Min=d2;d2MinIdx=6;}
        }
        if (d0_012>0 && d1_012>0 && d2_012>0) {
            d2 = (d0_012*s00+d1_012*s01+d2_012*s02)/d_012;
            if (d2<d2Min) {d2Min=d2;d2MinIdx=7;}
        }
        d2 = (s33);
        if (d2 < d2Min) {
            d2Min = d2;
            d2MinIdx = 8;
        }
        if (d0_03>0 && d3_03>0) {
            d2 = (d0_03*s00+d3_03*s03)/d_03;
            if (d2<d2Min) {d2Min=d2;d2MinIdx=9;}
        }
        if (d1_13>0 && d3_13>0) {
            d2 = (d1_13*s11+d3_13*s13)/d_13;
            if (d2<d2Min) {d2Min=d2;d2MinIdx=10;}
        }
        if (d0_013>0 && d1_013>0 && d3_013>0) {
            d2 = (d0_013*s00+d1_013*s01+d3_013*s03)/d_013;
            if (d2<d2Min) {d2Min=d2;d2MinIdx=11;}
        }
        if (d2_23>0 && d3_23>0) {
            d2 = (d2_23*s22+d3_23*s23)/d_23;
            if (d2<d2Min) {d2Min=d2;d2MinIdx=12;}
        }
        if (d0_023>0 && d2_023>0 && d3_023>0) {
            d2 = (d0_023*s00+d2_023*s02+d3_023*s03)/d_023;
            if (d2<d2Min) {d2Min=d2;d2MinIdx=13;}
        }
        if (d1_123>0 && d2_123>0 && d3_123>0) {
            d2 = (d1_123*s11+d2_123*s12+d3_123*s13)/d_123;
            if (d2<d2Min) {
                d2Min=d2;
                d2MinIdx=14;
            }
        }
        if (d0_0123>0 && d1_0123>0 && d2_0123>0 && d3_0123>0) {
            d2 = (d0_0123*s00+d1_0123*s01+d2_0123*s02+d3_0123*s03)/d_0123;
            if (d2<d2Min) {
                d2MinIdx=15;
            }
        }
        if (d2MinIdx==1) {
            corner[0].lambda=1;
            corner[1].lambda=0;
            corner[2].lambda=0;
            corner[3].lambda=0;
            return;
        }
        if (d2MinIdx==2) {
            corner[0].lambda=0;
            corner[1].lambda=1;
            corner[2].lambda=0;
            corner[3].lambda=0;
            return;
        }
        if (d2MinIdx==3) {
            float sumInv = 1/d_01;
            corner[0].lambda=d0_01*sumInv;
            corner[1].lambda=d1_01*sumInv;
            corner[2].lambda=0;
            corner[3].lambda=0;
            return;
        }
        if (d2MinIdx==4) {
            corner[0].lambda=0;
            corner[1].lambda=0;
            corner[2].lambda=1;
            corner[3].lambda=0;
            return;
        }
        if (d2MinIdx==5) {
            float sumInv = 1/d_02;
            corner[0].lambda=d0_02*sumInv;
            corner[1].lambda=0;
            corner[2].lambda=d2_02*sumInv;
            corner[3].lambda=0;
            return;
        }
        if (d2MinIdx==6) {
            float sumInv = 1/d_12;
            corner[0].lambda=0;
            corner[1].lambda=d1_12*sumInv;
            corner[2].lambda=d2_12*sumInv;
            corner[3].lambda=0;
            return;
        }
        if (d2MinIdx==7) {
            float sumInv = 1/d_012;
            corner[0].lambda=d0_012*sumInv;
            corner[1].lambda=d1_012*sumInv;
            corner[2].lambda=d2_012*sumInv;
            corner[3].lambda=0;
            return;
        }
        if (d2MinIdx==8) {
            corner[0].lambda=0;
            corner[1].lambda=0;
            corner[2].lambda=0;
            corner[3].lambda=1;
            return;
        }
        if (d2MinIdx==9) {
            float sumInv = 1/d_03;
            corner[0].lambda=d0_03*sumInv;
            corner[1].lambda=0;
            corner[2].lambda=0;
            corner[3].lambda=d3_03*sumInv;
            return;
        }
        if (d2MinIdx==10) {
            float sumInv = 1/d_13;
            corner[0].lambda=0;
            corner[1].lambda=d1_13*sumInv;
            corner[2].lambda=0;
            corner[3].lambda=d3_13*sumInv;
            return;
        }
        if (d2MinIdx==11) {
            float sumInv = 1/d_013;
            corner[0].lambda=d0_013*sumInv;
            corner[1].lambda=d1_013*sumInv;
            corner[2].lambda=0;
            corner[3].lambda=d3_013*sumInv;
            return;
        }
        if (d2MinIdx==12) {
            float sumInv = 1/d_23;
            corner[0].lambda=0;
            corner[1].lambda=0;
            corner[2].lambda=d2_23*sumInv;
            corner[3].lambda=d3_23*sumInv;
            return;
        }
        if (d2MinIdx==13) {
            float sumInv = 1/d_023;
            corner[0].lambda=d0_023*sumInv;
            corner[1].lambda=0;
            corner[2].lambda=d2_023*sumInv;
            corner[3].lambda=d3_023*sumInv;
            return;
        }
        if (d2MinIdx==14) {
            float sumInv = 1/d_123;
            corner[0].lambda=0;
            corner[1].lambda=d1_123*sumInv;
            corner[2].lambda=d2_123*sumInv;
            corner[3].lambda=d3_123*sumInv;
            return;
        }
        if (d2MinIdx==15) {
            float sumInv = 1/d_0123;
            corner[0].lambda=d0_0123*sumInv;
            corner[1].lambda=d1_0123*sumInv;
            corner[2].lambda=d2_0123*sumInv;
            corner[3].lambda=d3_0123*sumInv;
            return;
        }
    }

    //! Add a corner_
    /*! There are no more than four corners allowed. */
    void add (const Corner& corner_)
  {
    newCorner() = corner_;
    ++nrCorners;
  }

    //! Returns the next free \c corner entry
    Corner& newCorner()
    {
        for (auto & i : corner) if (!i.isValid()) return i;
        assert(false);
        return corner[0];
    }

    //! Returns, whether the pair \c aIdx, bIdx is a vertex of the tetrahedron
    bool isVertex (int aIdx, int bIdx) const
    {
        if (aIdx==corner[0].aIdx && bIdx==corner[0].bIdx) return true;
        if (aIdx==corner[1].aIdx && bIdx==corner[1].bIdx) return true;
        if (aIdx==corner[2].aIdx && bIdx==corner[2].bIdx) return true;
        if (aIdx==corner[3].aIdx && bIdx==corner[3].bIdx) return true;
        return false;
    }

};

#endif
