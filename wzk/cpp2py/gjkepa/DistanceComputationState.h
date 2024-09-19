#ifndef DISTANCE_COMPUTATION_STATE_H
#define DISTANCE_COMPUTATION_STATE_H

#include <limits>
#include "Volume.h"
#include "Simplex.h"
using namespace std;

/*!
    The state of the distance computation algorithm considering a fixed pair of volumes
    An object of this class is used for four things: a) Remember the input to the distance
    computation, i.e. which two volumes were involved. b) Remember which 4 vertices in both
    volumes were the one actually considered during one iteration. c) Remember this state
    across calls (time steps) to speed up by incrementally searching d) Return the result on
    the lower distance bound obtained as well as on the closest connection found so far (i.e.
    the actual points on both volumes.
*/

static int counter=0;


class DistanceComputationState : public Simplex{
public:
    //! Which body was volume a and b of the computation
    /*! This indices refer to some global volume list in their concrete
          usage here on \c CollisionTest::body.
      */
    int aVolumeId, bVolumeId;

    //! Index of the frame in which the comparison is performed
    /*! In the concrete usage in \c CollisionTest this is the least common ancestor
        of both bodies in the kinematic tree.  */
    int frameId;

    //! Index of the two effective volumes involved
    /*! \c bfiA is the effective volume of body \c aVolumeId in frame \c frameId,
        \c bfiB is the effective volume of body \c bVolumeId in frame \c frameId
      */
    int bfiA, bfiB;

    //! Whether to check this pair of volumes
    bool doCheckIt;
  
    //! The points on volume A and B that form the closest pair (found so far)
    /*! In \c frameId coordinates. See \c compute() */
    Vector pointOnA, pointOnB;
  
    //! The smallest distance bound found so far.
    /*! See \c compute() */
    float distanceBound{};

    //! The numerical epsilon used in comparing, whether lower and upper bound are equal
    /*! This is set by \c computeEps based on the norm of the affected volumes. */
    float eps;
  
    //! Empty object
    DistanceComputationState ():
            Simplex(), aVolumeId (-1), bVolumeId (-1), frameId(-1), bfiA(-1), bfiB(-1), doCheckIt(false), eps(0)
    {}

    //! Constructor initializing the configuration time data
    DistanceComputationState (int aVolumeId, int bVolumeId, int frameId, bool doCheckIt=true):
            Simplex(), aVolumeId (aVolumeId), bVolumeId (bVolumeId), frameId(frameId), bfiA(-1), bfiB(-1), doCheckIt(doCheckIt), eps(0)
    {}

    //! Subtracts \c changeA and \c changeB from \c distanceBound
    void updateBound (float changeA, float changeB) {
        distanceBound -= changeA + changeB;
    }

    //! Compute the distance (bound)
    /*! If the routine is called with the default parameters it finds
        the closest pair of points between \c volA and \c volB. The
        resulting is stored in \c *this both, where \c pointOnA and \c
        pointOnB are the respective closest points on both bodies and \c
        distanceBound is their distance.

        \c volA[corner[i].aIdx] and \c volB[corner[i].bIdx] (for all i
        for which \c corner[i].lambda>0) are the vertices spanning the
        faces/edges/vertices that are closest. They are convex combined
        according to the lambda values and modified according to \c
        volA.radius and \c volB.radius. This information is saved for
        the next call, because it provides a starting point for the GJK
        algorithm even if the objects have moved slightly (so actual
        coordinates would not be valid any more).

        The values for \c iteration and \c stopAbove define when to stop
        the iterative GJK computation before the final result has been
        reached. It is stopped either after \c iterations or when it has
        established a lower bound on the distance that is \c >=stopAbove
        (but always performing at least one iteration). In this case
        \c pointA and \c pointB are valid points of \c volA and \c volB,
        the vertex info is valid and \c distanceBound is a lower bound
        on the actual distance but \c distanceBound may be smaller than
        the distance between \c pointA and \c pointB being an upper bound.

        This feature is particularly useful when looking for the
        overall closest pair of points among some set of volumes. Then
        ones a upper bound is found by choosing a pair of points any distance
        computation between two volumes can be aborted if the lower bound
        is above this upper bound.

        The implementation is mainly a wrapper that calls \c
        computeWithoutRadius for computing the distance without taking
        \c volA.radius and \c volB.radius into account and adapting \c
        distanceBound, \c pointOnA, and \c pointOnB accordingly.
        */
    void compute (const Volume& volA, const Volume& volB, int iterations, float stopAbove)
    {
        assert (volA.isValid() && volB.isValid());
        if (eps == 0)
            setEpsByNorm(max(volA.maxNorm(), volB.maxNorm()));

        float radius_sum = volA.radius + volB.radius;
        computeWithoutRadius (volA, volB, iterations, stopAbove + radius_sum);

        // Adapt \c pointOnA and \c pointOnB by moving both inwards by \c volA.radius resp. volB.radius
        distanceBound -= radius_sum;
        Vector delta = pointOnB - pointOnA;

        float delta_norm = delta.twoNorm();
        pointOnA += volA.radius/delta_norm * delta;
        pointOnB -= volB.radius/delta_norm * delta;
        /*
        if (delta.twoNorm() > radius_sum) {
            pointOnA += volA.radius/n * delta;
            pointOnB -= volB.radius/n * delta;
        }
        else if (radius_sum > 0)  { // We choose a volA.radius:volB.radius mixture as the ambiguous intersection point
            pointOnA = pointOnB = pointOnA + (volA.radius) / radius_sum * delta;
        }
        */
    }

    //! Computes \c pointOnA and \c pointOnB from \c corner[i].lambda
    void computePointOnAB (const Volume& volA, const Volume& volB)
    {
        pointOnA = Vector();
        pointOnB = Vector();
        for (int i=0; i<nrCorners; i++){
            pointOnA += corner[i].lambda*volA[corner[i].aIdx];
            pointOnB += corner[i].lambda*volB[corner[i].bIdx];
        }
    }

    //! Recomputes \c corner[].v
    /*! It loads the original coordinates from \c volA and \c volB according
        to \c corner[].aIdx and \c corner[].bIdx. This has to be done
    after one time step where the robot moved but the corner indices
    are still a good first guess.

    Before doing so it checks, whether \c corner[idx] still has
    valid indices with respect to \c volA and \c volB If one of
    the up to four vertices corresponds to an invalid index the
    whole tetrahedron is reset to the containing only the first
    vertex of \c volA and \c volB.
    */
    void recomputeDifferences (const Volume& volA, const Volume& volB)
        {
            assert (!volA.empty() && !volB.empty());
            int nA = (int) volA.size();
            int nB = (int) volB.size();

            switch (nrCorners)
            {// runs through all cases and returns unless one corner doesn't exist anymore
                case 4:
                    if (corner[3].aIdx>=0 && corner[3].aIdx<nA && corner[3].bIdx>=0 && corner[3].bIdx<nB)
                        corner[3].v = volB[corner[3].bIdx] - volA[corner[3].aIdx];
                    else
                        break;
                case 3:
                    if (corner[2].aIdx>=0 && corner[2].aIdx<nA && corner[2].bIdx>=0 && corner[2].bIdx<nB)
                        corner[2].v = volB[corner[2].bIdx] - volA[corner[2].aIdx];
                    else
                        break;
                case 2:
                    if (corner[1].aIdx>=0 && corner[1].aIdx<nA && corner[1].bIdx>=0 && corner[1].bIdx<nB)
                        corner[1].v = volB[corner[1].bIdx] - volA[corner[1].aIdx];
                    else
                        break;
                case 1:
                    if (corner[0].aIdx>=0 && corner[0].aIdx<nA && corner[0].bIdx>=0 && corner[0].bIdx<nB)
                    {
                        corner[0].v = volB[corner[0].bIdx] - volA[corner[0].aIdx];
                        return; //return after successfully updating down to corner 0
                    }
                    else
                        break;
            }
            corner[1] = corner[2] = corner[3] = Corner();
            corner[0] = Corner (0, 0, volB[0]-volA[0]);
            corner[0].v = volB[corner[0].bIdx] - volA[corner[0].aIdx];
            nrCorners = 1;
        }

    //! Sets eps to a suitable value, if the max norm of all involved volumes is \c norm
    void setEpsByNorm (float norm)
    {
        eps = 10*sqrt(numeric_limits<float>::epsilon())*norm*norm;
    }


protected:
    //! Compute the distance without taking \c volA.radius and \c volB.radius into account
    /*! See \c compute for a description of the parameter. */
    void computeWithoutRadius (const Volume& volA, const Volume& volB, int iterations, float stopAbove)
    {
        counter++;
        recomputeDifferences (volA, volB);
        int ctr=0;
        while (true) { // break inside

            closestPoint();
            removeAll0Lambdas();
            computePointOnAB(volA, volB);

            // Evaluate projected distance and support points along direction dir
            Vector dir = pointOnB - pointOnA;
            int sPAIdx, sPBIdx;
            float maxScpA, maxScpB;
            volA.support(sPAIdx, maxScpA, dir);
            volB.support(sPBIdx, maxScpB, -dir);
            float d = -(maxScpA+maxScpB); //! Caution + not -, because the directions are opposite
            if (d>0)
                distanceBound = d/dir.twoNorm();
            else
                distanceBound = 0;

            if (ctr>=iterations || (ctr>0 && distanceBound>=stopAbove) || (distanceBound*dir.twoNorm()>dir.sqrTwoNorm()-eps))  // premature abort
                break;
            if (isVertex (sPAIdx, sPBIdx)) { // Has converged
                assert (distanceBound*dir.twoNorm()>dir.sqrTwoNorm()-eps); // very short
                break;
            }

            // Add the support points as a K-point to the K-tetrahedron
            add (Corner (sPAIdx, sPBIdx, volB[sPBIdx]-volA[sPAIdx]));
            ctr++;
        }
    }
};

/*
#define EPA_TOLERANCE 0.0001
#define EPA_TOLERANCE2 0.000001
#define EPA_MAX_NUM_FACES 64
#define EPA_MAX_NUM_LOOSE_EDGES 32
#define EPA_MAX_NUM_ITERATIONS 64
Vector EPA(const Vector& a, const Vector& b, const Vector& c, const Vector& d, Volume* volA, Volume* volB){
    Vector faces[EPA_MAX_NUM_FACES][4];  // Array of faces, each with 3 vertices and a normal

    //Init with final simplex from GJK
    faces[0][0] = a;
    faces[0][1] = b;
    faces[0][2] = c;
    faces[0][3] = normalize(cross(b-a, c-a)); //ABC
    faces[1][0] = a;
    faces[1][1] = c;
    faces[1][2] = d;
    faces[1][3] = normalize(cross(c-a, d-a)); //ACD
    faces[2][0] = a;
    faces[2][1] = d;
    faces[2][2] = b;
    faces[2][3] = normalize(cross(d-a, b-a)); //ADB
    faces[3][0] = b;
    faces[3][1] = d;
    faces[3][2] = c;
    faces[3][3] = normalize(cross(d-b, c-b)); //BDC

    int num_faces=4;
    int closest_face;

    for(int iterations=0; iterations<EPA_MAX_NUM_ITERATIONS; iterations++){
        //Find face that's closest to origin
        float min_dist = dot(faces[0][0], faces[0][3]);
        closest_face = 0;
        for(int i=1; i<num_faces; i++){
            float dist = dot(faces[i][0], faces[i][3]);
            if(dist<min_dist){
                min_dist = dist;
                closest_face = i;
            }
        }

        //search normal to face that's closest to origin
        Vector search_dir = faces[closest_face][3];
        Vector p = volB->support(search_dir) - volA->support(-search_dir);

        if(dot(p, search_dir)-min_dist<EPA_TOLERANCE){
            //Convergence (new point is not significantly further from origin)
            return faces[closest_face][3]*dot(p, search_dir); //dot vertex with normal to resolve collision along normal!
        }

        Vector loose_edges[EPA_MAX_NUM_LOOSE_EDGES][2]; //keep track of edges we need to fix after removing faces
        int num_loose_edges = 0;

        //Find all triangles that are facing p
        for(int i=0; i<num_faces; i++)
        {
            if(dot(faces[i][3], p-faces[i][0] )>0) //triangle i faces p, remove it
            {
                //Add removed triangle's edges to loose edge list.
                //If it's already there, remove it (both triangles it belonged to are gone)
                for(int j=0; j<3; j++) //Three edges per face
                {
                    Vector current_edge[2] = {faces[i][j], faces[i][(j+1)%3]};
                    bool found_edge = false;
                    for(int k=0; k<num_loose_edges; k++) //Check if current edge is already in list
                    {
                        if(loose_edges[k][1]==current_edge[0] && loose_edges[k][0]==current_edge[1]){
                            //Edge is already in the list, remove it
                            //THIS ASSUMES EDGE CAN ONLY BE SHARED BY 2 TRIANGLES (which should be true)
                            //THIS ALSO ASSUMES SHARED EDGE WILL BE REVERSED IN THE TRIANGLES (which
                            //should be true provided every triangle is wound CCW)
                            loose_edges[k][0] = loose_edges[num_loose_edges-1][0]; //Overwrite current edge
                            loose_edges[k][1] = loose_edges[num_loose_edges-1][1]; //with last edge in list
                            num_loose_edges--;
                            found_edge = true;
                            k=num_loose_edges; //exit loop because edge can only be shared once
                        }
                    }//endfor loose_edges

                    if(!found_edge){ //add current edge to list
                        // assert(num_loose_edges<EPA_MAX_NUM_LOOSE_EDGES);
                        if(num_loose_edges>=EPA_MAX_NUM_LOOSE_EDGES) break;
                        loose_edges[num_loose_edges][0] = current_edge[0];
                        loose_edges[num_loose_edges][1] = current_edge[1];
                        num_loose_edges++;
                    }
                }

                //Remove triangle i from list
                faces[i][0] = faces[num_faces-1][0];
                faces[i][1] = faces[num_faces-1][1];
                faces[i][2] = faces[num_faces-1][2];
                faces[i][3] = faces[num_faces-1][3];
                num_faces--;
                i--;
            }//endif p can see triangle i
        }//endfor num_faces

        //Reconstruct polytope with p added
        for(int i=0; i<num_loose_edges; i++)
        {
            // assert(num_faces<EPA_MAX_NUM_FACES);
            if(num_faces>=EPA_MAX_NUM_FACES) break;
            faces[num_faces][0] = loose_edges[i][0];
            faces[num_faces][1] = loose_edges[i][1];
            faces[num_faces][2] = p;
            faces[num_faces][3] = normalize(cross(loose_edges[i][0]-loose_edges[i][1], loose_edges[i][0]-p));

            //Check for wrong normal to maintain CCW winding
            //in case dot result is only slightly < 0 (because origin is on face)
            if(dot(faces[num_faces][0], faces[num_faces][3])+EPA_TOLERANCE2 < 0){
                Vector temp = faces[num_faces][0];
                faces[num_faces][0] = faces[num_faces][1];
                faces[num_faces][1] = temp;
                faces[num_faces][3] = -faces[num_faces][3];
            }
            num_faces++;
        }
    } //End for iterations
    printf("EPA did not converge\n");
    //Return most recent closest point
    return faces[closest_face][3] * dot(faces[closest_face][0], faces[closest_face][3]);
}

*/
#endif
