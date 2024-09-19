#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <iostream>
#include <vector>

#include <stack>
#include <ctime>

#include "Volume.h"
#include "DistanceComputationState.h"

std::vector<DistanceComputationState> collision_pairs;
std::vector<int> idx_a;
std::vector<int> idx_b;


PyObject * init_collision_pairs_py(PyObject * self, PyObject * args, PyObject * kwargs) {

    // initialize buffer for arguments
    PyObject * a;
    PyObject * b;
    int n;

    Py_buffer a_v;
    Py_buffer b_v;

    static const char * keywords[] = {"a", "b", "n", NULL};
    PyArg_ParseTupleAndKeywords(args, kwargs, "OOi", const_cast<char **>(keywords),
                                &a, &b, &n);

    PyObject_GetBuffer(a, &a_v, PyBUF_SIMPLE);
    PyObject_GetBuffer(b, &b_v, PyBUF_SIMPLE);

    // main
    int * a_p = (int *)a_v.buf;
    int * b_p = (int *)b_v.buf;

    collision_pairs.clear();
    for (int i=0; i<n; ++i){
        int ai = a_p[i];
        int bi = b_p[i];

        DistanceComputationState dcs(ai, bi, 0, true);

        collision_pairs.push_back(dcs);
        idx_a.push_back(ai);
        idx_b.push_back(bi);
    }

    // release buffer
    PyBuffer_Release(&a_v);
    PyBuffer_Release(&b_v);
    Py_RETURN_NONE;
}


PyObject * hull_hull_py(PyObject * self, PyObject * args, PyObject * kwargs) {

    // initialize buffer for arguments
    PyObject *xa, *xb, *ab, *c;
    float ra, rb;
    int na, nb, n;

    Py_buffer xa_v, xb_v, ab_v, c_v;

    static const char *keywords[] = {"xa", "ra", "na",
                                     "xb", "rb", "nb",
                                     "ab", "c", NULL};
    PyArg_ParseTupleAndKeywords(args, kwargs, "OfiOfiOO", const_cast<char **>(keywords),
                                &xa, &ra, &na,
                                &xb, &rb, &nb,
                                &ab, &c);

    PyObject_GetBuffer(xa, &xa_v, PyBUF_SIMPLE);
    PyObject_GetBuffer(xb, &xb_v, PyBUF_SIMPLE);
    PyObject_GetBuffer(ab, &ab_v, PyBUF_SIMPLE);
    PyObject_GetBuffer(c, &c_v, PyBUF_SIMPLE);

    // main
    Vector * xa_p = (Vector *)xa_v.buf;
    Vector * xb_p = (Vector *)xb_v.buf;

    float *  ab_p = (float *)ab_v.buf;
    float *  c_p = (float *)c_v.buf;


    Volume volA = Volume(xa_p, na, ra);
    Volume volB = Volume(xb_p, nb, rb);
    cout << volA[0] << "\n";
    cout << volA.radius << "\n";
    cout << volB[nb-1] << "\n";
    cout << volB.radius << "\n";

    DistanceComputationState dcs;
    dcs.compute(volA, volB, 1000, 100);
    cout << dcs.pointOnA << "\n";
    cout << dcs.pointOnB << "\n";
    // Write results
    ab_p[0] = dcs.pointOnA[0];
    ab_p[1] = dcs.pointOnA[1];
    ab_p[2] = dcs.pointOnA[2];
    ab_p[3] = dcs.pointOnB[0];
    ab_p[4] = dcs.pointOnB[1];
    ab_p[5] = dcs.pointOnB[2];
    c_p[0] = dcs.distanceBound;

    // release buffer
    PyBuffer_Release(&xa_v);
    PyBuffer_Release(&xb_v);
    PyBuffer_Release(&ab_v);
    PyBuffer_Release(&c_v);
    Py_RETURN_NONE;
}


PyMethodDef module_methods[] = {
        {"init_collision_pairs", (PyCFunction)init_collision_pairs_py, METH_VARARGS | METH_KEYWORDS, NULL},
        {"hull_hull", (PyCFunction)hull_hull_py, METH_VARARGS | METH_KEYWORDS, NULL},
        {NULL},
};


PyModuleDef module_def = {PyModuleDef_HEAD_INIT, "gjkepa", NULL, -1, module_methods};


extern "C" PyObject * PyInit_gjkepa() {
    PyObject * module = PyModule_Create(&module_def);
    return module;
}


