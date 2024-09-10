#define PY_SSIZE_T_CLEAN

#include <Python.h>

#include "openGJK.h"

#include <stdio.h>
#include <stdlib.h>


// Functions
// ---------------------------------------------------------------------------------------------------------------------



PyObject * compute_minimum_dist_py(PyObject * self, PyObject * args, PyObject * kwargs) {
    // initialize buffer for arguments
    PyObject * p1;
    PyObject * p2;
    PyObject * c1;
    PyObject * c2;
    PyObject * s;
    PyObject * d;
    int n1;
    int n2;

    Py_buffer p1_v;
    Py_buffer p2_v;
    Py_buffer c1_v;
    Py_buffer c2_v;
    Py_buffer s_v;
    Py_buffer d_v;
    static const char * keywords[] = {"p1", "n1", "p2", "n2", "s", "c1", "c2", "d", NULL};
    PyArg_ParseTupleAndKeywords(args, kwargs, "OiOiOOOO", const_cast<char **>(keywords),
                                &p1, &n1, &p2, &n2, &s, &c1, &c2, &d);
    PyObject_GetBuffer(p1, &p1_v, PyBUF_SIMPLE);
    PyObject_GetBuffer(p2, &p2_v, PyBUF_SIMPLE);
    PyObject_GetBuffer(c1, &c1_v, PyBUF_SIMPLE);
    PyObject_GetBuffer(c2, &c2_v, PyBUF_SIMPLE);
    PyObject_GetBuffer(s, &s_v, PyBUF_SIMPLE);
    PyObject_GetBuffer(d, &d_v, PyBUF_SIMPLE);


    // main
    // ---
    gkSimplex simplex;
    simplex.nvrtx = 0;

    gkPolytope polytope1;
    polytope1.numpoints = n1;

    gkPolytope polytope2;
    polytope2.numpoints = n2;

    // TODO not sure if this copying is the most efficient way to do it
	polytope1.coord = (double **) malloc(polytope1.numpoints * sizeof(double *));
    for (int i = 0; i < polytope1.numpoints; ++i) {
		polytope1.coord[i] = (double *) malloc(3 * sizeof(double));
    }

    double (*buf1)[3] = (double(*)[3])p1_v.buf;
    for (int i = 0; i < polytope1.numpoints; ++i) {
        for (int j = 0; j < 3; ++j) {
            polytope1.coord[i][j] = buf1[i][j];
        }
    }

//    printf("\n");
//    for (int i = 0; i < polytope1.numpoints; ++i) {
//        for (int j = 0; j < 3; ++j) {
//            printf("%f ",polytope1.coord[i][j]);
//        }
//        printf("\n"),
//    }

    polytope2.coord = (double **) malloc(polytope2.numpoints * sizeof(double *));
    for (int i = 0; i < polytope2.numpoints; ++i) {
		polytope2.coord[i] = (double *) malloc(3 * sizeof(double));
    }

    double (*buf2)[3] = (double(*)[3])p2_v.buf;
    for (int i = 0; i < polytope2.numpoints; ++i) {
        for (int j = 0; j < 3; ++j) {
            polytope2.coord[i][j] = buf2[i][j];
        }
    }


    double distance = compute_minimum_distance(&polytope1, &polytope2, &simplex);


    // copy results to python
    ((double *)d_v.buf)[0] = distance;

    for (int j = 0; j < 3; ++j) {
        ((double *)c1_v.buf)[j] = polytope1.s[j];
        ((double *)c2_v.buf)[j] = polytope2.s[j];
    }


    // ---




    PyBuffer_Release(&p1_v);
    PyBuffer_Release(&p2_v);
    PyBuffer_Release(&c1_v);
    PyBuffer_Release(&c2_v);
    PyBuffer_Release(&s_v);
    PyBuffer_Release(&d_v);
    Py_RETURN_NONE;
}


PyObject * trytry_py(PyObject * self, PyObject * args, PyObject * kwargs) {
    // initialize buffer for arguments
    PyObject * d;
    int n;

    Py_buffer p1_v;
    Py_buffer p2_v;
    Py_buffer c1_v;
    Py_buffer c2_v;
    Py_buffer s_v;
    Py_buffer d_v;
    printf("0 \n");
    static const char * keywords[] = {"d", "n", NULL};
    PyArg_ParseTupleAndKeywords(args, kwargs, "Oi", const_cast<char **>(keywords),
                                &d, &n);
    PyObject_GetBuffer(d, &d_v, PyBUF_SIMPLE);


    printf("1 \n");

     //((double *)d_v.buf)[0] = (double) a;
    ((double *)d_v.buf)[0] = (double) 1.23;

    printf("6 \n");
    // ---


    // release buffer
    PyBuffer_Release(&d_v);
    Py_RETURN_NONE;
}



// Expose to Python
// ---------------------------------------------------------------------------------------------------------------------
PyMethodDef module_methods[] = {
    {"compute_minimum_dist", (PyCFunction)compute_minimum_dist_py, METH_VARARGS | METH_KEYWORDS, NULL},
    {"trytry", (PyCFunction)trytry_py, METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL},
};

PyModuleDef module_def = {PyModuleDef_HEAD_INIT, "wzkopenGJK", NULL, -1, module_methods};

extern "C" PyObject * PyInit_wzkopenGJK() {
    PyObject * module = PyModule_Create(&module_def);
    return module;
}

